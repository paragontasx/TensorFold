"""tools/zig/triton_aot_manifest.py on the host: constexprs, PTX checks, ambiguity, the CLI, aot_pack.py reading it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "zig"))
sys.path.insert(0, str(ROOT / "zig" / "tests" / "cuda" / "nemotron"))

import aot_pack  # noqa: E402
import triton_aot_manifest as aot  # noqa: E402

PTX = """
.visible .entry _k(
	.param .u64 .ptr .global .align 1 _k_param_0,
	.param .u32 _k_param_1,
	.param .f32 _k_param_2,
	.param .u64 .ptr .global .align 1 _k_param_3,
	.param .u64 .ptr .global .align 1 _k_param_4
)
"""
SIGNATURE = {"X": "*bf16", "n": "i32", "f": "fp32", "one": "constexpr", "SCALE": "constexpr", "BLOCK": "constexpr"}


def kernel(hash_: str, cubin: str, data: bytes, attrs: dict | None = None) -> dict:
    """A dump entry as Recorder.dump writes it for SIGNATURE: `one` folded to 1, SCALE and BLOCK constexprs."""

    return {"function": "pkg.mod._k", "name": "_k", "hash": hash_, "signature": SIGNATURE,
            "constexprs": {"one": 1, "SCALE": aot.const_json(1.25), "BLOCK": 64},
            "attrs": {"X": [["tt.divisibility", 16]], "one": [], "SCALE": [], "BLOCK": []} if attrs is None else attrs,
            "abi": aot.abi("_k", SIGNATURE, PTX), "cubin": cubin, "cubin_sha256": hashlib.sha256(data).hexdigest(),
            "metadata": {"name": "_k", "num_warps": 4, "num_ctas": 1, "shared": 0, "global_scratch_size": 0,
                         "global_scratch_align": 1, "profile_scratch_size": 0, "launch_pdl": False},
            "target": "GPUTarget(backend='cuda', arch=121, warp_size=32)"}


def test_constexprs_keep_ints_and_bools_and_carry_floats_as_fp32_bits():
    assert aot.const_json(64) == 64 and aot.const_json(True) is True
    assert aot.const_json(1.25) == {"fp32_bits": "3fa00000"}
    assert aot.const_json(0.08838834764831845) == {"fp32_bits": "3db504f3"}      # 1 / sqrt(128), rounded to fp32
    assert "repr" in aot.const_json("bf16")


def test_abi_names_the_runtime_arguments_then_the_scratch_pointers():
    names = [p["name"] for p in aot.abi("_k", SIGNATURE, PTX)]
    assert names == ["X", "n", "f", "global_scratch", "profile_scratch"]


def test_abi_refuses_a_parameter_count_or_a_width_that_does_not_fit():
    with pytest.raises(ValueError, match="parameters"):
        aot.abi("_k", {**SIGNATURE, "m": "i32"}, PTX)
    with pytest.raises(ValueError, match="PTX takes"):
        aot.abi("_k", {"X": "*bf16", "n": "fp64", "f": "fp32"}, PTX)
    with pytest.raises(ValueError, match="no .entry"):
        aot.abi("_other", SIGNATURE, PTX)


def test_two_hashes_of_one_specialization_are_refused():
    a, b = kernel("a" * 64, "/aot/triton/A/_k.cubin", b"a"), kernel("b" * 64, "/aot/triton/B/_k.cubin", b"b")
    with pytest.raises(ValueError, match="differ only"):
        aot.check_unambiguous([a, b])
    aot.check_unambiguous([a, kernel("c" * 64, "/aot/triton/C/_k.cubin", b"c", attrs={})])   # X unaligned: distinct


def test_cli_writes_cache_relative_cubins_and_checks_their_hashes(tmp_path, monkeypatch):
    cache = tmp_path / "triton"
    (cache / "H1").mkdir(parents=True)
    (cache / "H1" / "_k.cubin").write_bytes(b"cubin")
    launches, out = tmp_path / "launches.json", tmp_path / "manifest.json"
    argv = ["triton_aot_manifest.py", "--cache", str(cache), "--launches", str(launches), "--mount", "/aot/triton",
            "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    good = kernel("h" * 64, "/aot/triton/H1/_k.cubin", b"cubin")
    launches.write_text(json.dumps({"triton": "3.7.1", "kernels": [good]}))
    assert aot.main() == 0
    assert json.loads(out.read_text())["kernels"][0]["cubin"] == "H1/_k.cubin"
    launches.write_text(json.dumps({"kernels": [kernel("h" * 64, "/aot/triton/H1/_k.cubin", b"other")]}))
    with pytest.raises(SystemExit, match="changed"):
        aot.main()
    launches.write_text(json.dumps({"kernels": [kernel("h" * 64, "/elsewhere/H1/_k.cubin", b"cubin")]}))
    with pytest.raises(SystemExit, match="not under"):
        aot.main()


def test_aot_pack_reads_the_manifest(tmp_path, monkeypatch):
    cache = tmp_path / "triton"
    (cache / "H1").mkdir(parents=True)
    (cache / "H1" / "_k.cubin").write_bytes(b"cubin")
    manifest, jit, out = tmp_path / "manifest.json", tmp_path / "jit.json", tmp_path / "kernels"
    entry = {**kernel("h" * 64, "/aot/triton/H1/_k.cubin", b"cubin"), "cubin": "H1/_k.cubin"}
    manifest.write_text(json.dumps({"kernels": [entry]}))
    jit.write_text(json.dumps({"pkg.mod._k": {"params": list(SIGNATURE), "do_not_specialize": ["n"], "no_align": []}}))
    argv = ["aot_pack.py", "--manifest", str(manifest), "--cache", str(cache), "--jit", str(jit), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert aot_pack.main() == 0
    (k,) = json.loads((out / "aot.json").read_text())["kernels"]
    assert (out / "cubins" / f"{'h' * 64}.cubin").read_bytes() == b"cubin"
    assert k["fn"] == "_k" and k["num_warps"] == 4
    assert [(p["name"], p["div16"], p["nospec"]) for p in k["params"]] == [("X", True, False), ("n", False, True),
                                                                          ("f", False, False)]
    assert k["consts"] == {"one": {"int": 1}, "SCALE": {"f32": 0x3FA00000}, "BLOCK": {"int": 64}}

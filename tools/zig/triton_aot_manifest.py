#!/usr/bin/env python3
"""The Triton variants an oracle run compiled, as the manifest aot_pack.py reads: capture.py's recorder and a CLI."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

GENERATOR = "tools/zig/triton_aot_manifest.py"
METADATA = ("name", "num_warps", "num_ctas", "shared", "global_scratch_size", "global_scratch_align",
            "profile_scratch_size", "launch_pdl")
SCRATCH = ("global_scratch", "profile_scratch")
PTX_TYPES = {"i32": {".u32", ".s32", ".b32"}, "u32": {".u32", ".s32", ".b32"},
             "i64": {".u64", ".s64", ".b64"}, "u64": {".u64", ".s64", ".b64"},
             "fp32": {".f32", ".b32"}, "fp64": {".f64", ".b64"},
             "fp16": {".b16", ".f16", ".u16"}, "bf16": {".b16", ".u16"},
             "i1": {".pred", ".u8", ".u16", ".u32"}, "i8": {".u8", ".s8", ".b8", ".u16"}}


def const_json(value):
    """A constexpr as aot_pack.py reads it: ints and bools as they are, floats by their fp32 bits."""

    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return {"fp32_bits": f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"}
    return {"repr": repr(value)}                    # aot_pack.py refuses it: no Zig form


def ptx_params(ptx: str, name: str) -> list[tuple[str, str]]:
    """(PTX type, PTX name) of each parameter of entry ``name``."""

    entry = re.search(r"\.entry\s+" + re.escape(name) + r"\s*\((.*?)\)", ptx, re.DOTALL)
    if entry is None:
        raise ValueError(f"no .entry {name} in the PTX")
    out = []
    for item in entry.group(1).split(","):
        words = item.split()
        if not words:
            continue
        if words[0] != ".param":
            raise ValueError(f"{name}: unexpected PTX parameter {item.strip()!r}")
        out.append((words[1], words[-1]))
    return out


def abi(kernel: str, signature: dict[str, str], ptx: str) -> list[dict]:
    """The entry's parameters, runtime arguments then Triton's two scratch pointers, once count and widths fit."""

    runtime = [n for n, t in signature.items() if t != "constexpr"]
    params = ptx_params(ptx, kernel)
    if len(params) != len(runtime) + len(SCRATCH):
        raise ValueError(f"{kernel}: the PTX has {len(params)} parameters for {len(runtime)} runtime arguments "
                         f"and {len(SCRATCH)} scratch pointers")
    out = []
    for (ptx_type, ptx_name), name in zip(params, runtime + list(SCRATCH)):
        want = {".u64"} if name in SCRATCH or signature[name].startswith("*") else PTX_TYPES.get(signature[name])
        if want is None:
            raise ValueError(f"{kernel}: no PTX width known for {name}, a {signature[name]}")
        if ptx_type not in want:
            what = signature.get(name, "a scratch pointer")
            raise ValueError(f"{kernel}: {name} is {what} but the PTX takes {ptx_type}")
        out.append({"name": name, "ptx": ptx_name, "ptx_type": ptx_type})
    return out


def variant(fn, kernel) -> dict:
    """One CompiledKernel as a manifest entry, its cubin still at the cache path inside the oracle's container."""

    src, md = kernel.src, kernel.metadata

    def name_of(path):
        if len(path) != 1:
            raise ValueError(f"{src.name}: nested parameter {path} has no Zig form")
        return fn.arg_names[path[0]]

    cubins = [p for c, p in kernel.metadata_group.items() if c.endswith(".cubin")]
    if len(cubins) != 1:
        raise ValueError(f"{src.name} {kernel.hash}: {len(cubins)} cubins in its metadata group")
    data = Path(cubins[0]).read_bytes()
    if data != kernel.asm["cubin"]:
        raise ValueError(f"{src.name} {kernel.hash}: the cache's cubin is not the one Triton loaded")
    signature = dict(src.signature)
    return {
        "function": f"{fn.fn.__module__}.{fn.fn.__qualname__}",
        "name": src.name,
        "hash": kernel.hash,
        "signature": signature,
        "constexprs": {name_of(p): const_json(v) for p, v in src.constants.items()},
        "attrs": {name_of(p): v for p, v in src.attrs.items()},
        "abi": abi(md.name, signature, kernel.asm["ptx"]),
        "cubin": cubins[0],
        "cubin_sha256": hashlib.sha256(data).hexdigest(),
        "metadata": {f: getattr(md, f) for f in METADATA if hasattr(md, f)},
        "target": str(md.target),
    }


def check_unambiguous(kernels: list[dict]) -> None:
    """Refuse two hashes the Zig matcher reads as one: same function, constexprs and runtime specialization."""

    seen: dict[str, str] = {}
    for k in kernels:
        runtime = [n for n, t in k["signature"].items() if t != "constexpr"]
        div16 = {n for n, v in k["attrs"].items() if v}
        key = json.dumps([k["name"], k["constexprs"], [(n, k["signature"][n], n in div16) for n in runtime]],
                         sort_keys=True)
        if seen.get(key, k["hash"]) != k["hash"]:
            raise ValueError(f"{k['name']}: variants {seen[key]} and {k['hash']} differ only where the Zig matcher "
                             "does not look (num_warps, stages)")
        seen[key] = k["hash"]


class Recorder:
    """capture.py's recorder: nothing is hooked during the run; dump lists what every JITFunction compiled."""

    def install(self) -> Recorder:
        return self

    def scope(self, name: str, detail: bool = False):
        return contextlib.nullcontext()

    def wrap(self, ext, names, label: str) -> None:
        return None

    def dump(self, path) -> None:
        import gc

        import triton
        from triton.compiler.compiler import CompiledKernel
        from triton.runtime.jit import JITFunction

        kernels, skipped = [], []
        for fn in gc.get_objects():
            if not isinstance(fn, JITFunction):
                continue
            for device, cache in fn.device_caches.items():
                for kernel in cache[0].values():
                    if isinstance(kernel, CompiledKernel):
                        kernels.append(variant(fn, kernel))
                    else:
                        skipped.append(f"{fn.fn.__qualname__} on device {device}: {type(kernel).__name__}")
        kernels.sort(key=lambda k: (k["name"], k["hash"]))
        check_unambiguous(kernels)
        Path(path).write_text(json.dumps({"generator": GENERATOR, "triton": triton.__version__, "kernels": kernels,
                                          "skipped": skipped}, indent=1) + "\n")
        print(f"[triton_aot_manifest] {len(kernels)} variants of {len({k['name'] for k in kernels})} functions "
              f"-> {path} ({len(skipped)} cache entries skipped)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="the oracle run's Triton cache on this host")
    ap.add_argument("--launches", required=True, help="the file Recorder.dump wrote")
    ap.add_argument("--mount", required=True, help="where the oracle's container mounted that cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache, mount = Path(a.cache), a.mount.rstrip("/") + "/"
    dump = json.loads(Path(a.launches).read_text())
    kernels = []
    for k in dump["kernels"]:
        if not k["cubin"].startswith(mount):
            raise SystemExit(f"{k['name']} {k['hash']}: cubin {k['cubin']} is not under {mount}")
        rel = k["cubin"][len(mount):]
        if hashlib.sha256((cache / rel).read_bytes()).hexdigest() != k["cubin_sha256"]:
            raise SystemExit(f"{cache / rel} changed since the oracle run")
        kernels.append({**k, "cubin": rel})
    check_unambiguous(kernels)
    Path(a.out).write_text(json.dumps({"generator": GENERATOR, "triton": dump.get("triton"), "kernels": kernels},
                                      indent=1) + "\n")
    print(f"{len(kernels)} variants of {len({k['name'] for k in kernels})} functions -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

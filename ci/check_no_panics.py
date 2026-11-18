#!/usr/bin/env python3
"""Fail if a branded benchmark kernel contains any `call` instruction.

The branded kernels are leaf loops: every bounds check is discharged by the types, so the
optimized code should never call into a panic handler. A bounds check that sneaks back in
shows up as a `call` to `panic_bounds_check` (or a slice-index panic) and fails this check.
"""
import json
import re
import subprocess
import sys

KERNELS = [
    "gather::branded",
    "spmv::branded",
    "spmv::branded_reassoc",
    "spmv::branded_adaptive",
    "union_find::branded",
    "stencil::branded",
]


def bench_executable() -> str:
    out = subprocess.run(
        ["cargo", "bench", "-p", "branded-index-bench", "--no-run", "--message-format=json"],
        check=True, capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") \
                and "bench" in msg["target"]["kind"]:
            return msg["executable"]
    sys.exit("bench executable not found")


def main() -> None:
    exe = bench_executable()
    symbols = subprocess.run(["nm", "-C", "-S", exe], check=True, capture_output=True, text=True).stdout
    failed = False
    for kernel in KERNELS:
        name = f"branded_index_bench::{kernel}"
        hits = [l.split(maxsplit=3) for l in symbols.splitlines()
                if re.fullmatch(re.escape(name) + r"(::<.*>)?", l.split(maxsplit=3)[-1])]
        if not hits:
            print(f"MISSING  {name}")
            failed = True
            continue
        for addr, size, _kind, sym in hits:
            start, end = int(addr, 16), int(addr, 16) + int(size, 16)
            asm = subprocess.run(
                ["objdump", "-d", "--no-show-raw-insn", f"--start-address={start:#x}",
                 f"--stop-address={end:#x}", exe],
                check=True, capture_output=True, text=True,
            ).stdout
            calls = [l.strip() for l in asm.splitlines() if re.search(r"\scall", l)]
            status = "ok      " if not calls else "FAIL    "
            print(f"{status} {sym} ({int(size, 16)} bytes, {len(calls)} calls)")
            for c in calls:
                print(f"           {c}")
            failed |= bool(calls)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

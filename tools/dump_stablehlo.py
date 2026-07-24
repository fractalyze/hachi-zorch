# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Trace verifier round functions to StableHLO fixture files for zorch-fv.

Each fixture is the exact `jax.jit(fn).lower(*args).as_text()` output for one
round function at one config; zorch-fv extracts these to Lean and pins their
hashes next to the equivalence proofs. Run from a venv with the Fractal JAX
build (field dtypes such as `jnp.babybear` must be native).

Usage:
    python tools/dump_stablehlo.py OUTPUT_DIR
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys

import jax
import jax.numpy as jnp


def _dump(name: str, fn, args, out_dir: pathlib.Path) -> None:
    text = jax.jit(fn).lower(*args).as_text()
    path = out_dir / f"{name}.stablehlo.txt"
    path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    ops = sorted(set(re.findall(r"stablehlo\.\w+", text)))
    print(f"{name}: sha256={digest}")
    print(f"{name}: ops={ops}")


def main(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fixture set grows with the verifier: one entry per round function per
    # proved config. Empty until hachi_zorch.verifier lands its first round.
    fixtures: list[tuple[str, object, tuple]] = []
    for name, fn, args in fixtures:
        _dump(name, fn, args, out_dir)
    if not fixtures:
        print("no fixtures registered yet; see hachi_zorch/verifier.py")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures"))

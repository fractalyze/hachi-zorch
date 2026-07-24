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
    from hachi_zorch.verifier import (
        ROUND_DEG_ALPHA,
        eval_split,
        linf_norm_check,
        paired_round_check,
        round_deg_zero,
    )

    bb = jnp.babybear
    b = 2
    msg_len = (round_deg_zero(b) + 1) + (ROUND_DEG_ALPHA + 1)
    # One fixture per pure kernel per proved config; names carry the config.
    fixtures = [
        (
            f"paired_round_check_b{b}_babybear",
            paired_round_check,
            (jnp.zeros((2,), bb), jnp.zeros((msg_len,), bb), jnp.zeros((), bb)),
        ),
        (
            "eval_split_nl2_nh2_babybear",
            eval_split,
            (jnp.zeros((4, 4), bb), jnp.zeros((2,), bb), jnp.zeros((2,), bb)),
        ),
        (
            "linf_norm_check_babybear",
            lambda resp: linf_norm_check(resp, 2013265921, 3),
            (jnp.zeros((8,), jnp.uint32),),
        ),
    ]
    for name, fn, args in fixtures:
        _dump(name, fn, args, out_dir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures"))

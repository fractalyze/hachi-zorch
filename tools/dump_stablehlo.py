# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Trace verifier round functions to StableHLO fixture files for zorch-fv.

Each fixture is the exact `frx.jit(fn).lower(*args).as_text()` output for one
round function at one config; zorch-fv extracts these to Lean and pins their
hashes next to the equivalence proofs. Run from a venv with the Fractal JAX
build, which is what makes Hachi's base field -- a parametric prime field,
since no curated family satisfies its congruence -- traceable at all.

Usage:
    python tools/dump_stablehlo.py OUTPUT_DIR
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys

import frx
import frx.numpy as fnp
import numpy as np

from hachi_zorch import field
from hachi_zorch.verifier import (
    ROUND_DEG_ALPHA,
    eval_split,
    linf_norm_check,
    paired_round_check,
    paired_rounds_check,
    round_deg_zero,
)


def _dump(name: str, fn, args, out_dir: pathlib.Path) -> None:
    text = frx.jit(fn).lower(*args).as_text()
    path = out_dir / f"{name}.stablehlo.txt"
    path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    ops = sorted(set(re.findall(r"stablehlo\.\w+", text)))
    print(f"{name}: sha256={digest}")
    print(f"{name}: ops={ops}")


def main(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fq = field.base_field()
    k = field.EXT_DEGREE

    def elems(*leading: int):
        """Zeros of the given shape in `F_q^k` elements -- the trailing
        coefficient axis is the element, never a protocol dimension."""
        return fnp.zeros((*leading, k), fq)

    b = 2
    msg_len = (round_deg_zero(b) + 1) + (ROUND_DEG_ALPHA + 1)
    chain_msg_len = (round_deg_zero(1) + 1) + (ROUND_DEG_ALPHA + 1)
    # One fixture per pure kernel per proved config; names carry the config.
    fixtures = [
        (
            f"paired_round_check_b{b}_fq4",
            paired_round_check,
            (elems(2), elems(msg_len), elems()),
        ),
        (
            "eval_split_nl2_nh2_fq4",
            eval_split,
            (elems(4, 4), elems(2), elems(2)),
        ),
        (
            "linf_norm_check_fq",
            lambda resp: linf_norm_check(resp, field.MODULUS, 3),
            (fnp.zeros((8,), np.uint32),),
        ),
        (
            "paired_rounds_check_m2_b1_fq4",
            paired_rounds_check,
            (elems(2), elems(2, chain_msg_len), elems(2)),
        ),
    ]
    for name, fn, args in fixtures:
        _dump(name, fn, args, out_dir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures"))

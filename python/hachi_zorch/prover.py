# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Honest test-vector prover for the paired sum-check rounds.

Unverified by design (see CLAUDE.md): exists so verifier tests and fixtures
have honest transcripts to accept and tampered ones to reject. Challenges are
inputs, matching the verifier kernels' Fiat-Shamir seam.

Summands are products of two multilinears, which pins both round-polynomial
degrees to 2 (the `b = 1` gadget config: `roundDegZero 1 = 2 = roundDegAlpha`).
Tables are dense hypercube evaluations over `F_q^k`, variable 0 on the
least-significant index bit; binding variable 0 at `t` sends
`T[y] -> T[2y] + t*(T[2y+1]-T[2y])`, so the round polynomial of a two-factor
product has closed-form ascending coefficients

    c0 = sum A0*B0,  c1 = sum (A0*dB + B0*dA),  c2 = sum dA*dB

with `A0[y] = A[2y]`, `dA[y] = A[2y+1]-A[2y]` (likewise `B`), summed over the
unbound tail -- no interpolation step, hence exact in any field dtype.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array

from hachi_zorch import field


def _split(table: Array) -> tuple[Array, Array]:
    even, odd = table[0::2], table[1::2]
    return even, odd - even


def _round_coeffs(a: Array, bb: Array) -> Array:
    a0, da = _split(a)
    b0, db = _split(bb)
    return fnp.stack(
        [
            field.sum_elements(field.mul(a0, b0)),
            field.sum_elements(field.mul(a0, db) + field.mul(b0, da)),
            field.sum_elements(field.mul(da, db)),
        ]
    )


def _bind(table: Array, t: Array) -> Array:
    even, diff = _split(table)
    return even + field.mul(t, diff)


def prove_paired_sumcheck(
    f0: tuple[Array, Array], fa: tuple[Array, Array], rs: Array
) -> tuple[Array, Array]:
    """Round messages for the paired sum-check of two product summands.

    `f0` / `fa` are the factor tables of the `H0`- and `H_alpha`-summands
    (each `(2^m0, k)`), `rs` the `(m0, k)` shared challenges. Returns
    `(claims, msgs)`: the initial claim pair `[sum f0, sum fa]` and the
    `(m0, 6, k)` coefficient rows (`g0` then `ga`, ascending)."""
    a, b = f0
    c, d = fa
    claims = fnp.stack(
        [
            field.sum_elements(field.mul(a, b)),
            field.sum_elements(field.mul(c, d)),
        ]
    )
    rows = []
    for i in range(rs.shape[0]):
        rows.append(fnp.concatenate([_round_coeffs(a, b), _round_coeffs(c, d)]))
        a, b = _bind(a, rs[i]), _bind(b, rs[i])
        c, d = _bind(c, rs[i]), _bind(d, rs[i])
    return claims, fnp.stack(rows)


def final_evals(
    f0: tuple[Array, Array], fa: tuple[Array, Array], rs: Array
) -> Array:
    """`[prod of f0 factors, prod of fa factors]` fully bound at `rs` -- what
    the final reduced claim must equal for an honest transcript."""
    a, b = f0
    c, d = fa
    for i in range(rs.shape[0]):
        a, b, c, d = (_bind(x, rs[i]) for x in (a, b, c, d))
    return fnp.stack([field.mul(a[0], b[0]), field.mul(c[0], d[0])])

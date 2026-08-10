# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Hachi's field stack and the extension arithmetic the verifier runs on.

[NOZ26] separates three levels: the base prime field `F_q` holding committed
coefficients, the extension `F_q^k` where every verifier operation happens, and
the cyclotomic ring `R_q = Z_q[X]/(X^d + 1)` the prover commits in. Only the
first two appear here -- the verification algorithm performs no ring
operations, which is what puts Hachi's verifier within reach of an equivalence
proof at all.

An `F_q^k` element is carried as `k` base-field coefficients on a trailing
axis, not as a native extension dtype. Two reasons, in order: the parametric
extension descriptors do not cross the frx frontend yet, so no native dtype
exists for this base field; and spelling the extension arithmetic out leaves
every base-field multiplication visible in the traced module, which is the
level zorch-fv's equivalence theorems are stated at.
"""

from __future__ import annotations

from typing import Any

import frx.numpy as fnp
import zk_dtypes
from frx import Array, lax

# [NOZ26] pins no concrete modulus. It requires `q = 5 (mod 8)` (Lemma 5 --
# the congruence fixes how `X^d + 1` factors over `Z_q`, and with it the
# invertibility of challenge differences that the extractor needs) and sizes q
# at roughly 2^32. This is the largest prime below 2^32 meeting the congruence.
MODULUS = 2**32 - 99

# Challenges must carry the full soundness margin, so the extension degree is
# `ceil(lambda / log2 q)` = 4 at `lambda = 128`.
EXT_DEGREE = 4

# `X^4 - 2` is irreducible over `F_q`: `q = 5 (mod 8)` makes 2 a quadratic
# non-residue, and a non-residue's multiplicative order carries the same
# 2-adic valuation as `q - 1` (namely 2, again by the congruence). Those are
# exactly Lidl-Niederreiter 3.75's conditions for `x^4 - c`, whose remaining
# requirement `q = 1 (mod 4)` the congruence also implies.
EXT_NON_RESIDUE = 2


def base_field() -> Any:
    """The `F_q` dtype: a parametric prime field, since no curated family
    satisfies Hachi's congruence on q."""
    return zk_dtypes.prime_field(MODULUS, "mont")


def one(dtype: Any) -> Array:
    """`1` in `F_q^k`, as the coefficient vector `(1, 0, ..., 0)`.

    Built by concatenation rather than a scatter into zeros: a scatter would
    put `stablehlo.scatter` in the trace of every monomial basis, for a
    constant.
    """
    return fnp.concatenate(
        [fnp.ones((1,), dtype), fnp.zeros((EXT_DEGREE - 1,), dtype)]
    )


def sum_elements(x: Array, axis: int = 0) -> Array:
    """Sum `F_q^k` elements along `axis`, keeping the coefficient axis.

    `lax.reduce` rather than `fnp.sum`: the numpy-level reduction routes its
    result dtype through an integer-width promotion guard that recognizes only
    the curated field families, and raises on a parametric descriptor.
    """
    return lax.reduce(x, fnp.zeros((), x.dtype), lax.add, (axis,))


def mul(a: Array, b: Array) -> Array:
    """Product in `F_q^k = F_q[X]/(X^k - EXT_NON_RESIDUE)`, broadcasting over
    the leading axes.

    Schoolbook, with the wrap-around terms (those of degree `>= k`) folded back
    by the non-residue -- the reduction `X^k = EXT_NON_RESIDUE` applied once,
    which suffices because a product of two degree-`<k` polynomials has degree
    `< 2k`.
    """
    if a.shape[-1] != EXT_DEGREE or b.shape[-1] != EXT_DEGREE:
        raise ValueError(
            f"operands must carry {EXT_DEGREE} coefficients on the trailing "
            f"axis, got {a.shape[-1]} and {b.shape[-1]}"
        )
    nr = fnp.full((), EXT_NON_RESIDUE, a.dtype)
    a0, a1, a2, a3 = (a[..., i] for i in range(EXT_DEGREE))
    b0, b1, b2, b3 = (b[..., i] for i in range(EXT_DEGREE))
    c0 = a0 * b0 + nr * (a1 * b3 + a2 * b2 + a3 * b1)
    c1 = a0 * b1 + a1 * b0 + nr * (a2 * b3 + a3 * b2)
    c2 = a0 * b2 + a1 * b1 + a2 * b0 + nr * (a3 * b3)
    c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0
    return fnp.stack([c0, c1, c2, c3], axis=-1)


def eq(a: Array, b: Array) -> Array:
    """Equality of `F_q^k` elements: all `k` coefficients agree."""
    return fnp.all(a == b, axis=-1)

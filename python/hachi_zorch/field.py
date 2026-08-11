# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Hachi's field stack.

[NOZ26] separates three levels: the base prime field `F_q` holding committed
coefficients, the extension `F_q^k` where every verifier operation happens, and
the cyclotomic ring `R_q = Z_q[X]/(X^d + 1)` the prover commits in. Only the
first two appear here -- the verification algorithm performs no ring
operations, which is what puts Hachi's verifier within reach of an equivalence
proof at all.

Both are parametric `zk_dtypes` dtypes: no curated family satisfies Hachi's
congruence on `q`, so the fields are minted from their parameters. `F_q^k` is
a first-class element type end to end -- one `stablehlo.multiply` on an
`!field.ef` tensor is one extension product -- so the verifier kernels below
never spell the extension arithmetic out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zk_dtypes
from frx import Array

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
    """The `F_q` dtype."""
    return zk_dtypes.prime_field(MODULUS, "mont")


def ext_field() -> Any:
    """The `F_q^k` dtype -- every verifier value's type."""
    return zk_dtypes.extension_field(
        MODULUS, EXT_DEGREE, EXT_NON_RESIDUE, "mont"
    )


def from_coeffs(rows) -> np.ndarray:
    """`F_q^k` elements from integer coefficient rows, ascending per row.

    Goes through a base-field buffer and reinterprets it, the same way a
    transcript reads `k` consecutive squeezes as one challenge: there is no
    host-side constructor taking a coefficient tuple directly.
    """
    flat = np.array([c for row in rows for c in row], dtype=base_field())
    return flat.view(ext_field())


def to_coeffs(x: Array) -> np.ndarray:
    """Coefficient rows of an `F_q^k` array -- the inverse of `from_coeffs`.

    Flattened before the reinterpret so a scalar (a reduction's result) reads
    as one row; numpy refuses to retype a 0-d array across itemsizes.
    """
    return np.asarray(x).reshape(-1).view(base_field()).reshape(-1, EXT_DEGREE)

# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Hachi verifier rounds -- spec-shaped duals of ArkLib's Hachi tree.

Each public name transcribes one verifier-side definition from ArkLib
`Commitments/Functional/Hachi/` and is named after it. The pure kernels
(`paired_round_check`, `eval_split`, `linf_norm_check`) take challenges as
arguments and are what `tools/dump_stablehlo.py` traces -- the Fiat-Shamir
seam the zorch-fv equivalence proofs cut at. `PairedSumcheckRound` wraps the
round kernel in zorch's `InnerVerifierRound` contract for use in a chain.

Round messages are coefficient-form, matching ArkLib's `RoundMsg F b`: the
wire carries the pair `(g0, ga)` concatenated as one array of
`(2b + 1) + 3` coefficients, and the round identity reads `g(0) = c_0`,
`g(1) = sum(c)` directly (`roundCheck` in `Hachi/Sumcheck/Rounds.lean`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from zorch.round import Round
from zorch.transcript import Transcript, sample_challenge

ROUND_DEG_ALPHA = 2  # degree of g_alpha, pinned by [NOZ26] Figure 6


def _eval_coeffs_unrolled(coeffs: Array, r: Array) -> Array:
    """`sum_i coeffs[i] * r**i`, unrolled. zorch's `eval_coeffs` carries the
    power-sum through a `lax.scan` to keep GPU kernel parameters O(1) in the
    degree; that emits `stablehlo.while`, and a control-flow-free trace is
    worth more to this lane than kernel-size headroom at degree <= 2b."""
    acc = coeffs[0]
    power = r
    for i in range(1, coeffs.shape[0]):
        acc = acc + coeffs[i] * power
        power = power * r
    return acc


def round_deg_zero(b: int) -> int:
    """Degree of g_0 for gadget base-decomposition width `b`: `roundDegZero b = 2b`."""
    return 2 * b


def paired_round_check(claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
    """One paired sum-check round: `roundCheck` + claim reduction, Figure 6.

    `claim` is `[target0, target_alpha]`; `msg` is `g0`'s then `ga`'s ascending
    coefficients concatenated (static split -- the caller's degree config fixes
    it); `r` is the shared scalar challenge. Returns
    `([g0(r), ga(r)], ok)` with `ok` the conjunction of both round identities
    `g(0) + g(1) == target`.
    """
    split = msg.shape[0] - (ROUND_DEG_ALPHA + 1)
    g0, ga = msg[:split], msg[split:]
    ok0 = claim[0] == g0[0] + jnp.sum(g0)
    oka = claim[1] == ga[0] + jnp.sum(ga)
    next_claim = jnp.stack([_eval_coeffs_unrolled(g0, r), _eval_coeffs_unrolled(ga, r)])
    return next_claim, ok0 & oka


def monomial_basis(x: Array) -> Array:
    """Monomial tensor basis `mb(x)`: entry `i` is `prod_j x_j^{bit_j(i)}`,
    little-endian (bit 0 = first variable), matching `CMlPolynomial.monomialBasis`."""
    basis = jnp.ones((1,), x.dtype)
    for j in range(x.shape[0]):
        basis = jnp.concatenate([basis, basis * x[j]])
    return basis


def eval_split(matrix: Array, xl: Array, xh: Array) -> Array:
    """Multilinear evaluation as the vector-matrix-vector product
    `mb(xl) . (matrix @ mb(xh))` -- `evalSplit` in `Hachi/EvalSplit.lean`.

    `matrix` is the `2^nl x 2^nh` reshape of the coefficient vector, rows
    indexed by the low (first) variables, columns by the high (last) ones.
    """
    return monomial_basis(xl) @ (matrix @ monomial_basis(xh))


def linf_norm_check(response: Array, modulus: int, bound: int) -> Array:
    """Centered infinity-norm check on an opening response, given as canonical
    residues in `[0, modulus)`: every entry's centered representative has
    absolute value `<= bound`. Load-bearing for unique gadget decomposition
    (binding of the packed claim); dropping it admits multiple valid unpackings.
    """
    centered = jnp.minimum(response, modulus - response)
    return jnp.all(centered <= bound)


def paired_rounds_check(
    claim: Array, msgs: Array, rs: Array
) -> tuple[Array, Array]:
    """A chain of paired sum-check rounds — the loop of [NOZ26] Figure 7,
    mirroring ArkLib's `roundsChain` recursion over the guarded append.

    `msgs` is `(m0, msg_len)` (one coefficient row per round), `rs` the
    `(m0,)` shared challenges. The loop is a Python loop over the static
    leading dimension, so the trace stays control-flow-free. Returns the
    final reduced claim pair and the conjunction of every round identity.
    """
    ok = None
    for i in range(msgs.shape[0]):
        claim, ok_i = paired_round_check(claim, msgs[i], rs[i])
        ok = ok_i if ok is None else ok & ok_i
    return claim, ok


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["b", "challenge_limbs"],
)
@dataclass(frozen=True)
class PairedSumcheckRound(Round):
    """Verifier for one paired Hachi sum-check round, in zorch's
    `InnerVerifierRound` contract: observes the coefficient pair, samples the
    shared challenge, and delegates the check to `paired_round_check`."""

    b: int
    challenge_limbs: int = 1

    def __post_init__(self) -> None:
        if self.b < 1:
            raise ValueError("gadget base width b must be >= 1")
        if self.challenge_limbs < 1:
            raise ValueError("challenge_limbs must be >= 1")

    @property
    def msg_len(self) -> int:
        return (round_deg_zero(self.b) + 1) + (ROUND_DEG_ALPHA + 1)

    def __call__(
        self, claim: Array, msg: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array, Array]:
        if msg.shape[0] != self.msg_len:
            raise ValueError(
                f"round message must carry {self.msg_len} coefficients "
                f"(g0 deg {round_deg_zero(self.b)}, ga deg {ROUND_DEG_ALPHA}), "
                f"got {msg.shape[0]}"
            )
        transcript = transcript.observe(msg)
        transcript, r = sample_challenge(transcript, claim.dtype, self.challenge_limbs)
        next_claim, ok = paired_round_check(claim, msg, r)
        return next_claim, transcript, r, ok

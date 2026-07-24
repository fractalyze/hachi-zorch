# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure verifier kernels.

Completeness cases build honest values by construction (a message whose round
identities hold, an eval-split matrix checked against the direct monomial sum);
soundness cases tamper one value and expect rejection. All cases run the pure
kernels on the CPU backend -- no transcript, no GPU."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from hachi_zorch.verifier import (
    ROUND_DEG_ALPHA,
    eval_split,
    linf_norm_check,
    monomial_basis,
    paired_round_check,
    round_deg_zero,
)

BB = jnp.babybear


def _honest_msg(claim, g0_tail, ga_tail):
    """Coefficient vectors whose round identities hold: choose every
    coefficient but c_0 freely, then solve `c_0 + sum(c) == target` for c_0
    (i.e. `2*c_0 = target - sum(tail)`; babybear is odd-characteristic, so
    halving is exact)."""
    inv2 = jnp.array((2013265921 + 1) // 2, BB)  # 2^{-1} mod babybear prime
    c0_0 = (claim[0] - jnp.sum(g0_tail)) * inv2
    c0_a = (claim[1] - jnp.sum(ga_tail)) * inv2
    return jnp.concatenate(
        [jnp.stack([c0_0]), g0_tail, jnp.stack([c0_a]), ga_tail]
    )


class PairedRoundCheckTest(absltest.TestCase):
    def test_honest_message_accepts_and_reduces(self) -> None:
        b = 2
        claim = jnp.array([7, 11], BB)
        g0_tail = jnp.arange(1, round_deg_zero(b) + 1, dtype=BB)
        ga_tail = jnp.arange(3, 3 + ROUND_DEG_ALPHA, dtype=BB)
        msg = _honest_msg(claim, g0_tail, ga_tail)
        r = jnp.array(5, BB)

        next_claim, ok = paired_round_check(claim, msg, r)
        self.assertTrue(bool(ok))

        # The reduced claim is the pair of round-poly evaluations at r.
        split = round_deg_zero(b) + 1
        g0, ga = msg[:split], msg[split:]
        powers = [jnp.array(1, BB)]
        for _ in range(split - 1):
            powers.append(powers[-1] * r)
        direct0 = sum((g0[i] * powers[i] for i in range(split)), jnp.array(0, BB))
        directa = sum((ga[i] * powers[i] for i in range(ROUND_DEG_ALPHA + 1)), jnp.array(0, BB))
        self.assertTrue(bool(next_claim[0] == direct0))
        self.assertTrue(bool(next_claim[1] == directa))

    def test_tampered_coefficient_rejects(self) -> None:
        b = 2
        claim = jnp.array([7, 11], BB)
        g0_tail = jnp.arange(1, round_deg_zero(b) + 1, dtype=BB)
        ga_tail = jnp.arange(3, 3 + ROUND_DEG_ALPHA, dtype=BB)
        msg = _honest_msg(claim, g0_tail, ga_tail)
        tampered = msg.at[1].set(msg[1] + jnp.array(1, BB))

        _, ok = paired_round_check(claim, tampered, jnp.array(5, BB))
        self.assertFalse(bool(ok))

    def test_tampering_only_ga_rejects(self) -> None:
        b = 1
        claim = jnp.array([7, 11], BB)
        msg = _honest_msg(
            claim,
            jnp.arange(1, round_deg_zero(b) + 1, dtype=BB),
            jnp.arange(3, 3 + ROUND_DEG_ALPHA, dtype=BB),
        )
        tampered = msg.at[-1].set(msg[-1] + jnp.array(1, BB))
        _, ok = paired_round_check(claim, tampered, jnp.array(5, BB))
        self.assertFalse(bool(ok))


class EvalSplitTest(absltest.TestCase):
    def test_matches_direct_monomial_sum(self) -> None:
        nl, nh = 2, 2
        rng = np.random.default_rng(0)
        coeffs = jnp.array(rng.integers(0, 97, size=2 ** (nl + nh)), BB)
        xl = jnp.array(rng.integers(0, 97, size=nl), BB)
        xh = jnp.array(rng.integers(0, 97, size=nh), BB)

        # Direct evaluation: coefficient i weights prod_j x_j^{bit_j(i)},
        # little-endian across the concatenated point (xl first).
        x = jnp.concatenate([xl, xh])
        total = jnp.array(0, BB)
        for i in range(2 ** (nl + nh)):
            term = coeffs[i]
            for j in range(nl + nh):
                if (i >> j) & 1:
                    term = term * x[j]
            total = total + term

        # Row index = low bits (first variables), column = high bits.
        matrix = coeffs.reshape(2**nh, 2**nl).T
        self.assertTrue(bool(eval_split(matrix, xl, xh) == total))

    def test_monomial_basis_little_endian(self) -> None:
        x = jnp.array([3, 5], BB)
        mb = monomial_basis(x)
        expected = [1, 3, 5, 15]  # [1, x0, x1, x0*x1]
        for i, e in enumerate(expected):
            self.assertTrue(bool(mb[i] == jnp.array(e, BB)))


class LinfNormCheckTest(absltest.TestCase):
    MODULUS = 2013265921  # babybear prime; responses arrive as canonical residues

    def test_within_bound_accepts(self) -> None:
        response = jnp.array([0, 3, self.MODULUS - 3], jnp.uint32)
        self.assertTrue(bool(linf_norm_check(response, self.MODULUS, 3)))

    def test_exceeding_bound_rejects(self) -> None:
        response = jnp.array([0, 4, self.MODULUS - 3], jnp.uint32)
        self.assertFalse(bool(linf_norm_check(response, self.MODULUS, 3)))

    def test_negative_side_exceeding_rejects(self) -> None:
        response = jnp.array([0, 3, self.MODULUS - 4], jnp.uint32)
        self.assertFalse(bool(linf_norm_check(response, self.MODULUS, 3)))


if __name__ == "__main__":
    absltest.main()

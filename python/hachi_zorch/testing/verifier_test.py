# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure verifier kernels.

Completeness cases build honest values by construction (a message whose round
identities hold, an eval-split matrix checked against the direct monomial sum);
soundness cases tamper one value and expect rejection. Everything is over
`F_q^k`, so a "scalar" here is a `(k,)` coefficient vector. All cases run the
pure kernels on the CPU backend -- no transcript, no GPU.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hachi_zorch import field
from hachi_zorch.verifier import (
    ROUND_DEG_ALPHA,
    eval_split,
    linf_norm_check,
    monomial_basis,
    paired_round_check,
    paired_rounds_check,
    round_deg_zero,
)

F = field.base_field()
Q = field.MODULUS
K = field.EXT_DEGREE


def fq4(coeffs):
    return fnp.asarray(np.array(coeffs, dtype=F))


def base(values):
    """`F_q^k` elements embedded from the base field: `(v, 0, ..., 0)`."""
    return fq4([[v] + [0] * (K - 1) for v in values])


def rand(rng, n: int):
    return fq4(rng.integers(0, Q, size=(n, K)).tolist())


def _honest_msg(claim, g0_tail, ga_tail):
    """Coefficient vectors whose round identities hold: choose every
    coefficient but c_0 freely, then solve `c_0 + sum(c) == target` for c_0
    (i.e. `2*c_0 = target - sum(tail)`; q is odd, so halving is exact)."""
    inv2 = base([(Q + 1) // 2])[0]  # 2^{-1} mod q, embedded in F_q^k
    c0_0 = field.mul(claim[0] - field.sum_elements(g0_tail), inv2)
    c0_a = field.mul(claim[1] - field.sum_elements(ga_tail), inv2)
    return fnp.concatenate(
        [c0_0[None, :], g0_tail, c0_a[None, :], ga_tail]
    )


class PairedRoundCheckTest(absltest.TestCase):
    def test_honest_message_accepts_and_reduces(self) -> None:
        b = 2
        rng = np.random.default_rng(0)
        claim = base([7, 11])
        g0_tail = rand(rng, round_deg_zero(b))
        ga_tail = rand(rng, ROUND_DEG_ALPHA)
        msg = _honest_msg(claim, g0_tail, ga_tail)
        r = rand(rng, 1)[0]

        next_claim, ok = paired_round_check(claim, msg, r)
        self.assertTrue(bool(ok))

        # The reduced claim is the pair of round-poly evaluations at r,
        # recomputed here by an independent Horner-free power ladder.
        split = round_deg_zero(b) + 1
        g0, ga = msg[:split], msg[split:]
        powers = [field.one(F)]
        for _ in range(split - 1):
            powers.append(field.mul(powers[-1], r))
        direct0 = g0[0]
        for i in range(1, split):
            direct0 = direct0 + field.mul(g0[i], powers[i])
        directa = ga[0]
        for i in range(1, ROUND_DEG_ALPHA + 1):
            directa = directa + field.mul(ga[i], powers[i])
        self.assertTrue(bool(field.eq(next_claim[0], direct0)))
        self.assertTrue(bool(field.eq(next_claim[1], directa)))

    def test_tampered_coefficient_rejects(self) -> None:
        rng = np.random.default_rng(1)
        b = 2
        claim = base([7, 11])
        msg = _honest_msg(claim, rand(rng, round_deg_zero(b)), rand(rng, ROUND_DEG_ALPHA))
        tampered = msg.at[1].set(msg[1] + field.one(F))

        _, ok = paired_round_check(claim, tampered, rand(rng, 1)[0])
        self.assertFalse(bool(ok))

    def test_tampering_only_ga_rejects(self) -> None:
        rng = np.random.default_rng(2)
        b = 1
        claim = base([7, 11])
        msg = _honest_msg(claim, rand(rng, round_deg_zero(b)), rand(rng, ROUND_DEG_ALPHA))
        tampered = msg.at[-1].set(msg[-1] + field.one(F))
        _, ok = paired_round_check(claim, tampered, rand(rng, 1)[0])
        self.assertFalse(bool(ok))

    def test_tampering_a_single_extension_coefficient_rejects(self) -> None:
        # A tamper confined to one F_q coordinate of one F_q^k element must
        # still be caught: the accept bit compares all k coordinates.
        rng = np.random.default_rng(3)
        b = 1
        claim = base([7, 11])
        msg = _honest_msg(claim, rand(rng, round_deg_zero(b)), rand(rng, ROUND_DEG_ALPHA))
        tampered = msg.at[1, K - 1].set(msg[1, K - 1] + fnp.ones((), F))
        _, ok = paired_round_check(claim, tampered, rand(rng, 1)[0])
        self.assertFalse(bool(ok))


class EvalSplitTest(absltest.TestCase):
    def test_matches_direct_monomial_sum(self) -> None:
        nl, nh = 2, 2
        rng = np.random.default_rng(0)
        coeffs = rand(rng, 2 ** (nl + nh))
        xl = rand(rng, nl)
        xh = rand(rng, nh)

        # Direct evaluation: coefficient i weights prod_j x_j^{bit_j(i)},
        # little-endian across the concatenated point (xl first).
        x = fnp.concatenate([xl, xh])
        total = fnp.zeros((K,), F)
        for i in range(2 ** (nl + nh)):
            term = coeffs[i]
            for j in range(nl + nh):
                if (i >> j) & 1:
                    term = field.mul(term, x[j])
            total = total + term

        # Row index = low bits (first variables), column = high bits.
        matrix = coeffs.reshape(2**nh, 2**nl, K).transpose(1, 0, 2)
        self.assertTrue(bool(field.eq(eval_split(matrix, xl, xh), total)))

    def test_monomial_basis_little_endian(self) -> None:
        x = base([3, 5])
        mb = monomial_basis(x)
        expected = base([1, 3, 5, 15])  # [1, x0, x1, x0*x1]
        for i in range(4):
            self.assertTrue(bool(field.eq(mb[i], expected[i])))

    def test_monomial_basis_uses_the_extension_product(self) -> None:
        # With a genuinely non-base point, entry 3 must be the extension
        # product x0*x1 -- not a coordinatewise one.
        x = fq4([[0, 1, 0, 0], [0, 0, 0, 1]])  # X and X^3
        mb = monomial_basis(x)
        self.assertTrue(bool(field.eq(mb[3], field.mul(x[0], x[1]))))
        # X * X^3 = X^4 = NR.
        self.assertTrue(
            bool(field.eq(mb[3], base([field.EXT_NON_RESIDUE])[0]))
        )


class LinfNormCheckTest(absltest.TestCase):
    def test_within_bound_accepts(self) -> None:
        response = fnp.asarray(np.array([0, 3, Q - 3], np.uint32))
        self.assertTrue(bool(linf_norm_check(response, Q, 3)))

    def test_exceeding_bound_rejects(self) -> None:
        response = fnp.asarray(np.array([0, 4, Q - 3], np.uint32))
        self.assertFalse(bool(linf_norm_check(response, Q, 3)))

    def test_negative_side_exceeding_rejects(self) -> None:
        response = fnp.asarray(np.array([0, 3, Q - 4], np.uint32))
        self.assertFalse(bool(linf_norm_check(response, Q, 3)))


class PairedRoundsCheckTest(absltest.TestCase):
    def test_chain_accepts_and_threads_the_claim(self) -> None:
        # Two rounds built honestly one after the other: the second round's
        # target is the first round's reduced claim.
        rng = np.random.default_rng(4)
        b = 1
        claim = base([7, 11])
        rs = rand(rng, 2)
        rows = []
        running = claim
        for i in range(2):
            msg = _honest_msg(
                running, rand(rng, round_deg_zero(b)), rand(rng, ROUND_DEG_ALPHA)
            )
            rows.append(msg)
            running, ok = paired_round_check(running, msg, rs[i])
            self.assertTrue(bool(ok))
        msgs = fnp.stack(rows)

        final, ok = paired_rounds_check(claim, msgs, rs)
        self.assertTrue(bool(ok))
        self.assertTrue(bool(field.eq(final[0], running[0])))
        self.assertTrue(bool(field.eq(final[1], running[1])))


if __name__ == "__main__":
    absltest.main()

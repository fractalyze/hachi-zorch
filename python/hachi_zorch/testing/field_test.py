# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Unit tests for Hachi's field stack.

Two things are worth pinning here. That the chosen modulus satisfies the
congruence [NOZ26] needs, and hence that `X^k - EXT_NON_RESIDUE` really is
irreducible -- otherwise the extension dtype is not a field at all. And that
the dtype's arithmetic agrees with a polynomial multiply done in Python
integers: the verifier's every operation is now one extension op, so if the
dtype reduced by the wrong non-residue (or by a differently-encoded one) every
kernel would be silently wrong, and nothing else in this repo would notice.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hachi_zorch import field

Q = field.MODULUS
K = field.EXT_DEGREE
NR = field.EXT_NON_RESIDUE


def elems(rows):
    return fnp.asarray(field.from_coeffs(rows))


def ints(x) -> list[list[int]]:
    return [[int(v) for v in row] for row in field.to_coeffs(x)]


def ref_mul(a: list[int], b: list[int]) -> list[int]:
    """`a * b` in `Z_q[X]/(X^k - NR)`, in Python integers."""
    prod = [0] * (2 * K - 1)
    for i in range(K):
        for j in range(K):
            prod[i + j] = (prod[i + j] + a[i] * b[j]) % Q
    return [
        (prod[k] + NR * (prod[k + K] if k + K < len(prod) else 0)) % Q
        for k in range(K)
    ]


class ModulusTest(absltest.TestCase):
    def test_congruence_holds(self) -> None:
        self.assertEqual(Q % 8, 5)

    def test_non_residue_is_a_quadratic_non_residue(self) -> None:
        # Euler's criterion. This is the whole justification for X^k - 2 being
        # irreducible, so it is worth asserting rather than trusting.
        self.assertEqual(pow(NR, (Q - 1) // 2, Q), Q - 1)

    def test_two_adicity_is_two(self) -> None:
        # q = 5 (mod 8) forces v_2(q-1) = 2, which is what pins the order of a
        # non-residue and hence the irreducibility criterion.
        self.assertEqual(((Q - 1) & -(Q - 1)).bit_length() - 1, 2)


class ExtensionArithmeticTest(absltest.TestCase):
    def test_matches_python_reference_on_random_inputs(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(32):
            a = [int(v) for v in rng.integers(0, Q, size=K)]
            b = [int(v) for v in rng.integers(0, Q, size=K)]
            got = ints(elems([a]) * elems([b]))[0]
            self.assertEqual(got, ref_mul(a, b))

    def test_x_to_the_k_is_the_non_residue(self) -> None:
        # The defining relation of the extension: X^k = NR. A dtype reducing by
        # a Montgomery-encoded non-residue instead of a canonical one fails
        # exactly here.
        x = elems([[0, 1, 0, 0]])
        power = x
        for _ in range(K - 1):
            power = power * x
        self.assertEqual(ints(power)[0], [NR, 0, 0, 0])

    def test_one_is_the_identity(self) -> None:
        a = elems([[7, 11, 13, 17]])
        self.assertEqual(ints(a * fnp.ones((1,), a.dtype)), ints(a))

    def test_sum_adds_coefficientwise(self) -> None:
        v = elems([[1, 2, 3, 4], [10, 20, 30, 40], [Q - 1, 0, 0, 0]])
        self.assertEqual(ints(fnp.sum(v))[0], [10, 22, 33, 44])

    def test_equality_sees_every_coefficient(self) -> None:
        a = elems([[1, 2, 3, 4]])
        self.assertTrue(bool((a == elems([[1, 2, 3, 4]]))[0]))
        self.assertFalse(bool((a == elems([[1, 2, 3, 5]]))[0]))
        self.assertFalse(bool((a == elems([[9, 2, 3, 4]]))[0]))


class CoeffRoundTripTest(absltest.TestCase):
    def test_from_coeffs_inverts_to_coeffs(self) -> None:
        rows = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(ints(elems(rows)), rows)

    def test_element_count_is_row_count(self) -> None:
        # A (2, k) coefficient block is 2 extension elements, not 2*k of them:
        # the coefficient axis is the element, never a protocol dimension.
        self.assertEqual(elems([[1, 2, 3, 4], [5, 6, 7, 8]]).shape, (2,))


if __name__ == "__main__":
    absltest.main()

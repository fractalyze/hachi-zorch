# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Unit tests for Hachi's field stack.

Two things are worth pinning here: that the chosen modulus actually satisfies
the congruence [NOZ26] needs (and hence that `X^k - EXT_NON_RESIDUE` really is
irreducible, which is what makes the coefficient representation a field at
all), and that `field.mul` agrees with a polynomial multiply done in Python
integers -- the traced kernel's reduction is the thing zorch-fv will prove
against, so it must not be self-referentially tested.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from hachi_zorch import field

F = field.base_field()
Q = field.MODULUS
K = field.EXT_DEGREE


def fq4(coeffs):
    """An `F_q^k` element (or array of them) from integer coefficients."""
    return fnp.asarray(np.array(coeffs, dtype=F))


def to_ints(x) -> list[int]:
    return [int(c) for c in np.asarray(x)]


def ref_mul(a: list[int], b: list[int]) -> list[int]:
    """`a * b` in `Z_q[X]/(X^k - NR)`, in Python integers."""
    prod = [0] * (2 * K - 1)
    for i in range(K):
        for j in range(K):
            prod[i + j] = (prod[i + j] + a[i] * b[j]) % Q
    out = []
    for k in range(K):
        wrap = prod[k + K] if k + K < len(prod) else 0
        out.append((prod[k] + field.EXT_NON_RESIDUE * wrap) % Q)
    return out


class ModulusTest(absltest.TestCase):
    def test_congruence_holds(self) -> None:
        self.assertEqual(Q % 8, 5)

    def test_non_residue_is_a_quadratic_non_residue(self) -> None:
        # Euler's criterion. This is the whole justification for X^k - 2 being
        # irreducible, so it is worth asserting rather than trusting.
        self.assertEqual(pow(field.EXT_NON_RESIDUE, (Q - 1) // 2, Q), Q - 1)

    def test_two_adicity_is_two(self) -> None:
        # q = 5 (mod 8) forces v_2(q-1) = 2, which is what pins the order of a
        # non-residue and hence the irreducibility criterion.
        self.assertEqual(((Q - 1) & -(Q - 1)).bit_length() - 1, 2)


class MulTest(absltest.TestCase):
    def test_matches_python_reference_on_random_inputs(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(32):
            a = [int(v) for v in rng.integers(0, Q, size=K)]
            b = [int(v) for v in rng.integers(0, Q, size=K)]
            got = to_ints(field.mul(fq4(a), fq4(b)))
            self.assertEqual(got, ref_mul(a, b))

    def test_x_to_the_k_is_the_non_residue(self) -> None:
        # The defining relation of the extension: X^k = NR.
        x = fq4([0, 1, 0, 0])
        power = field.mul(x, x)
        for _ in range(K - 2):
            power = field.mul(power, x)
        self.assertEqual(to_ints(power), [field.EXT_NON_RESIDUE, 0, 0, 0])

    def test_one_is_the_identity(self) -> None:
        a = fq4([7, 11, 13, 17])
        self.assertEqual(to_ints(field.mul(a, field.one(F))), to_ints(a))

    def test_broadcasts_over_leading_axes(self) -> None:
        vec = fq4([[1, 2, 3, 4], [5, 6, 7, 8]])
        scalar = fq4([2, 0, 1, 0])
        got = field.mul(vec, scalar)
        for i in range(2):
            self.assertEqual(
                to_ints(got[i]), ref_mul(to_ints(vec[i]), to_ints(scalar))
            )

    def test_rejects_wrong_coefficient_count(self) -> None:
        with self.assertRaises(ValueError):
            field.mul(fq4([1, 2, 3]), fq4([1, 2, 3]))


class SumElementsTest(absltest.TestCase):
    def test_sums_along_the_element_axis(self) -> None:
        vec = fq4([[1, 2, 3, 4], [10, 20, 30, 40], [Q - 1, 0, 0, 0]])
        self.assertEqual(to_ints(field.sum_elements(vec)), [10, 22, 33, 44])

    def test_sums_a_chosen_axis(self) -> None:
        mat = fq4([[[1, 0, 0, 0], [2, 0, 0, 0]], [[3, 0, 0, 0], [4, 0, 0, 0]]])
        self.assertEqual(to_ints(field.sum_elements(mat, axis=1)[0]), [3, 0, 0, 0])
        self.assertEqual(to_ints(field.sum_elements(mat, axis=1)[1]), [7, 0, 0, 0])


class EqTest(absltest.TestCase):
    def test_all_coefficients_must_agree(self) -> None:
        a = fq4([1, 2, 3, 4])
        self.assertTrue(bool(field.eq(a, fq4([1, 2, 3, 4]))))
        self.assertFalse(bool(field.eq(a, fq4([1, 2, 3, 5]))))
        # A difference in the leading coefficient alone must also be caught.
        self.assertFalse(bool(field.eq(a, fq4([9, 2, 3, 4]))))


if __name__ == "__main__":
    absltest.main()

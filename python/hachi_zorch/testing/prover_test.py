# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Prover-verifier roundtrip for the paired sum-check chain.

Completeness: an honest transcript accepts every round and the final reduced
claim equals the fully-bound factor products. Soundness smoke: tampering any
coefficient row flips the conjunction."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from hachi_zorch.prover import final_evals, prove_paired_sumcheck
from hachi_zorch.verifier import paired_rounds_check

BB = jnp.babybear


def _random_instance(m0: int, seed: int):
    rng = np.random.default_rng(seed)
    tables = [jnp.array(rng.integers(0, 10007, size=2**m0), BB) for _ in range(4)]
    rs = jnp.array(rng.integers(1, 10007, size=m0), BB)
    return (tables[0], tables[1]), (tables[2], tables[3]), rs


class PairedRoundsRoundtripTest(absltest.TestCase):
    def _roundtrip(self, m0: int, seed: int) -> None:
        f0, fa, rs = _random_instance(m0, seed)
        claims, msgs = prove_paired_sumcheck(f0, fa, rs)
        final, ok = paired_rounds_check(claims, msgs, rs)
        self.assertTrue(bool(ok))

        expected = final_evals(f0, fa, rs)
        self.assertTrue(bool(final[0] == expected[0]))
        self.assertTrue(bool(final[1] == expected[1]))

    def test_two_rounds(self) -> None:
        self._roundtrip(2, seed=0)

    def test_four_rounds(self) -> None:
        self._roundtrip(4, seed=1)

    def test_tampered_row_rejects(self) -> None:
        f0, fa, rs = _random_instance(3, seed=2)
        claims, msgs = prove_paired_sumcheck(f0, fa, rs)
        one = jnp.array(1, BB)
        for row in range(msgs.shape[0]):
            tampered = msgs.at[row, 4].set(msgs[row, 4] + one)
            _, ok = paired_rounds_check(claims, tampered, rs)
            self.assertFalse(bool(ok), msg=f"tampered row {row} accepted")


if __name__ == "__main__":
    absltest.main()

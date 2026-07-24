# Copyright 2026 The Fractalyze Authors. SPDX-License-Identifier: Apache-2.0
"""Hachi verifier round functions -- spec-shaped duals of ArkLib's Hachi tree.

Every public function here transcribes one verifier-side definition from
ArkLib `Commitments/Functional/Hachi/` and is named after it. Challenges
arrive as arguments; transcript observe/sample happens in the caller (the
Fiat-Shamir seam the zorch-fv equivalence proofs cut at).
"""

from __future__ import annotations

from jax import Array


def sumcheck_round(claim: Array, msg: Array, r: Array) -> tuple[Array, Array]:
    """One sum-check round check: `Hachi/Sumcheck/Rounds.lean` dual.

    Returns (next_claim, ok). `ok` is the round identity `claim == s(0)+s(1)`;
    the reduced claim is `s(r)` for the round polynomial `s` carried by `msg`.
    """
    raise NotImplementedError


def quad_eval_claim_translation(claim: Array, point: Array) -> Array:
    """Map the field-level evaluation claim to the ring-level QuadEval claim:
    `Hachi/QuadEval/Bridge.lean` dual. Pure basis-change arithmetic."""
    raise NotImplementedError


def opening_norm_check(response: Array, bound: int) -> Array:
    """Shortness check on an opening response. Load-bearing for unique
    decomposition (binding of the packed claim); dropping a bound admits
    multiple valid unpackings."""
    raise NotImplementedError

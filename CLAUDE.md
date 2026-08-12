# Project context for Claude Code

Overview and the two-lane design live in [`README.md`](README.md) — start
there.

## Non-negotiables

- **The verifier mirrors ArkLib, not the other way around.** Every verifier
  round function is a one-to-one transcription of the corresponding definition
  under `ArkLib/Commitments/Functional/Hachi/`. When a shape choice trades
  speed against structural fidelity to the Lean definition, fidelity wins —
  this repo's verifier exists to be proved equivalent, in zorch-fv, to specs
  that mirror ArkLib.
- **Challenges are inputs, never computed inline.** Round functions take
  `(claim, msg, r)` with the challenge `r` as an argument; transcript
  observe/sample stays outside the round body. This is the Fiat-Shamir seam
  the equivalence proofs cut at — folding sampling into a round silently
  widens the proof obligation to the whole transcript.
- **Assemble zorch's blocks, never re-implement the spine.** Same rule as
  every `*-zorch` consumer: rounds, transcript, fold come from zorch; this
  repo adds only Hachi-specific pieces.
- **The prover is a test-vector generator.** It is unverified by design and
  must never be load-bearing for a soundness claim. Don't optimize it at the
  cost of verifier-lane clarity.

## Relationship to zorch-fv

zorch-fv consumes this repo's traced verifier modules as hash-pinned StableHLO
fixtures (`tools/dump_stablehlo.py` output). Any verifier behavior change must
be accompanied by a fixture re-dump, and lands only after zorch-fv's
equivalence proofs are updated to the new fixture hash.

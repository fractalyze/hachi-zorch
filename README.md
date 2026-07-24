# hachi-zorch

Hachi lattice-based multilinear polynomial commitments on the zorch spine,
with a formally-verifiable verifier lane.

Hachi ([NOZ26], [eprint 2026/156](https://eprint.iacr.org/2026/156)) is a
lattice-based (Module-SIS) multilinear PCS over extension fields, built on the
Greyhound inner-outer Ajtai commitment. Its verifier performs **no cyclotomic
ring operations** — plain extension-field arithmetic — which makes it the
scheme whose verifier connects most directly to the machine-checked soundness
proofs in [ArkLib](https://github.com/Verified-zkEVM/ArkLib)
(`ArkLib/Commitments/Functional/Hachi/`).

## The two lanes

- **Verifier (spec-shaped).** Written one-to-one against ArkLib's Hachi
  verifier definitions, structured for equivalence proofs in
  [fractalyze/zorch-fv](https://github.com/fractalyze/zorch-fv): pure round
  functions, challenges as inputs (transcript cut at the Fiat-Shamir seam),
  static shapes per config. Performance is explicitly not a goal of this lane;
  no fusion discipline applies.
- **Prover (unverified).** Exists to produce proofs for the verifier — ring
  NTTs, Ajtai commitments, gadget decompositions. Correctness by testing only;
  it appears in no soundness theorem.

## Verification contract

`tools/dump_stablehlo.py` traces the verifier's round functions to StableHLO
text. Those modules are the fixtures zorch-fv extracts to Lean and proves
equivalent to its specs. CI keeps the fixture hashes in sync; a verifier change
here invalidates the proofs there by construction.

## Status

Scaffold. Open items before the first end-to-end check, in dependency order:

1. Paper §3 (small-field packing) read → decide the packed claim-translation
   layout for BabyBear-class base fields.
2. Verifier round functions: sum-check rounds (zorch `Round` duals), QuadEval
   claim translation, norm checks on opening responses.
3. Test-vector prover for the same config.
4. zorch spine pin (Bazel MODULE wiring, matching the other `*-zorch`
   consumers).

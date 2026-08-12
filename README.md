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
equivalent to its specs, so a verifier change here invalidates the proofs there
by construction.

CI closes that loop rather than trusting it: the `fv` job traces this branch's
verifier into a zorch-fv checkout, requires the trace to match the fixtures
committed there, extracts them, and runs zorch-fv's `lake build`. A verifier
change that moves a fixture or breaks an equivalence proof fails before merge
instead of at the next dump.

A coupled change is therefore two pull requests, zorch-fv's first. That order
is the one that works: zorch-fv proves whatever it has committed without
consulting this repo, so it can carry a trace dumped from a branch here, while
the reverse order ships a verifier whose fixture is stale and is repairable
from neither side.

Extraction parses a fixture with MLIR rather than reading it as text, so the
bindings that print the assembly are the ones that read it back — both ends of
the contract run in the environment below.

## The field

[NOZ26] pins no concrete modulus, so this repo picks one: `q = 2^32 - 99`, the
largest prime below `2^32` satisfying the congruence `q = 5 (mod 8)` the scheme
requires. Verifier arithmetic is in `F_q^4` (degree `ceil(128 / log2 q)`), a
first-class element type end to end: one `stablehlo.multiply` is one extension
product, and no kernel spells the coefficient arithmetic out. See
`python/hachi_zorch/field.py` for the irreducibility argument behind `X^4 - 2`.

No curated field family satisfies Hachi's congruence, so both fields are
*parametric*. Tracing and extraction therefore both need a Fractal JAX build
carrying parametric prime and extension field support:

```sh
uv venv --python 3.11 .venv
uv pip install --index-url https://fractalyze.github.io/pypi/simple/ \
    --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
    frx frxlib zk-dtypes absl-py numpy typing_extensions
uv pip install --no-deps "pyzorch @ git+ssh://git@github.com/fractalyze/zorch@main"

PYTHONPATH=python python -m hachi_zorch.testing.verifier_test
PYTHONPATH=python python tools/dump_stablehlo.py ../zorch-fv/fixtures

# the same venv, from a zorch-fv checkout, re-extracts a fixture to Lean
python extractor/extract.py \
    fixtures/eval_split_nl2_nh2_fq4.stablehlo.txt \
    ZorchFv/Hachi/Extracted/eval_split_nl2_nh2_fq4.lean
```

## Status

Verifier kernels, test-vector prover, and fixtures are in place over the field
above: `paired_round_check` / `paired_rounds_check` (Figures 6 and 7),
`eval_split`, `monomial_basis`, and `linf_norm_check`. Open items, in
dependency order:

1. Paper §3 (small-field packing) → the packed claim-translation layout, and
   with it the partial-eval recombination and trace-map consistency kernels.
2. A transcript over `F_q`: the Fiat-Shamir seam is currently open (kernels
   take challenges as arguments, which is what the equivalence proofs want),
   and closing it needs sponge constants for this field. Until then there is
   no `VerifierRound` wrapper — an untestable one would be worse than none.
3. zorch spine pin (Bazel MODULE wiring, matching the other `*-zorch`
   consumers).

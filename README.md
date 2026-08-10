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

## The field

[NOZ26] pins no concrete modulus, so this repo picks one: `q = 2^32 - 99`, the
largest prime below `2^32` satisfying the congruence `q = 5 (mod 8)` the scheme
requires. Verifier arithmetic is in `F_q^4` (degree `ceil(128 / log2 q)`),
represented as four `F_q` coefficients on a trailing axis rather than a native
extension dtype — see `python/hachi_zorch/field.py` for why, and for the
irreducibility argument behind `X^4 - 2`.

No curated field family satisfies Hachi's congruence, so the base field is a
*parametric* prime field. Tracing therefore needs a Fractal JAX build with
parametric field support:

```sh
uv venv --python 3.11 .venv
uv pip install --index-url https://fractalyze.github.io/pypi/simple/ \
    --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
    frx frxlib zk-dtypes absl-py numpy typing_extensions
uv pip install --no-deps "pyzorch @ git+ssh://git@github.com/fractalyze/zorch@main"

PYTHONPATH=python python -m hachi_zorch.testing.verifier_test
PYTHONPATH=python python tools/dump_stablehlo.py ../zorch-fv/fixtures
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
4. A native `F_q^4` dtype in place of the coefficient representation, once
   parametric extension descriptors cross the frx frontend.

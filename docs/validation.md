# Validation and reference scores

This source release contains the benchmark evaluation code and container recipes.
It does not include model weights, benchmark data, private runtime configuration,
experiment logs, or the MMResearch controller.

## Local checks

The release candidate is checked with CPU regression fixtures and generated-bundle
consistency checks. These exercise split/role handling, score validation, grader
error classification, credential handling, and service contracts. They do not
constitute a GPU experiment, a complete image build, or a successful run of every
SWE-bench instance.

```bash
python3 scripts/sync_eval_bundles.py --check
python3 -m unittest discover -s tests/release -v
```

Some isolation tests require Linux static linking, a local Node TLS server, or Unix
socket peer-credential support and can be skipped on unsupported platforms. Inspect
the test report to distinguish passing checks from skipped checks.

## Container and MMSWE validation

Build the images and evaluate known positive and negative examples on the target
host before reporting model scores. MMSWE additionally requires compatible official
instance images, the pinned SWE-bench dependency, namespace/chroot support, and
working user/group transitions. An infrastructure failure is not a zero model
score and must not silently reduce the evaluation denominator. The alternative
controlled-service backend remains experimental; see its deployment guide.

## Verification results

The corrected release passed 121 of 126 CPU regression tests, with five platform-dependent
skips, plus 89 generated-bundle consistency checks. Python and shell syntax and
local documentation links were checked. A real GPU smoke generated a response
for a two-image development example. Native MMSWE grading passed a gold-patch
positive and two negative examples (empty and malformed patches).

Full container builds and browser-dependent instance grading require a compatible
Linux host and are not certified by these checks. Provide working cgroups,
namespace permissions, and the browsers required by official instance images;
keep browser sandbox protections enabled. Infrastructure failures invalidate the
run rather than becoming model scores. Small smoke checks do not reproduce the
complete 480-instance test or the full agent-training-verifier loop.

## Paper protocol and reference scores

The bundled `suite.yaml` and `src/eval/baselines.json` record the target-test
baselines and development/test sizes from Tables 2 and 7 of the paper. MMSWE uses
the official 100-instance development split and 480-instance test split. Its
published test baseline is 0.0229 (2.29%); the former 34-instance / 0.0588 development
calibration is no longer used by the release configuration.

`n_full` describes the combined data pool; `n_dev` and `n_eval` specify the separate
counts. Official MMSWE splits are never passed through the perception-task index
split. The recipe leaves `SWE_SPLIT` unset by default so generation and grading
both derive `val -> dev` and `eval -> test` from the effective evaluation role.

The selected benchmark is forwarded to the verifier bundle. Final verifier metrics
record task and split identity. The oracle compares raw metrics (before audit
reset), requiring the complete test denominator; its default accuracy tolerance of
0.00005 covers rounding of the four-decimal published fractions. Audit resets fail
without a matching baseline instead of silently substituting zero or the raw score.
For direct gate calls, set `GATE_METRICS` to the operator-owned verifier metrics file.

These checks catch common configuration errors, but do not certify dataset or model
identity. Freeze the checkpoint, dataset revision and selected IDs, media, and grader
configuration for a run. Use the same protocol for base and submitted models. A changed
instance set requires its own measured reference; do not relabel partial results as a
480-instance test score. The published numbers are reference metadata, not a claim
that all paper experiments have been rerun from this source package.

## Release corrections

Task selection, 24-hour budgets, guarded opt-in run resets, structured data
preparation, container dataset paths, contained image extraction and stale audit
verdict handling have dedicated CPU fixtures. These are implementation checks,
not new experimental results. Full public-data/container runs remain host-dependent.
The disabled Harbor live workspace judge always marks integrity as unknown; use
the separate operator audit gate and review incomplete checks.

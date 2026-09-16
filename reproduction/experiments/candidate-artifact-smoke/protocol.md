# Protocol: candidate project-account artifact viability

Status: pre-registered before execution.

## Question

Can the candidate `MorphServe/MorphServe` snapshot at `85c4fbf753b6eb47613bdd7ea423c38b48054544` build, install, import, and expose its core C++/Python mechanisms unmodified in the available environment?

## Prediction

The C++ extension may build after selecting the existing PyTorch 2.4/CUDA 12.4 environment, but the top-level Python package will fail unmodified because the release directory is `MorphServe`, setup declares `morphserve`, and source imports `swiftllm`. `EngineConfig`/API construction is also predicted to fail because required morphing fields have no CLI arguments. These are packaging/configuration failures, not evidence against the underlying mechanism.

## Frozen procedure

1. Verify the vendored manifest before touching runtime files.
2. Create an isolated `.venv` under `reproduction/` using Python 3.12 and install only pinned/cached dependencies needed for the probe. Do not modify the vendored artifact.
3. Attempt unmodified editable install from `vendor/author-morphserve`; save full stdout/stderr and exit code.
4. Attempt imports `MorphServe`, `morphserve`, and `swiftllm` with only that install; save results.
5. Build `vendor/author-morphserve/csrc` against the isolated environment; save compiler output, produced artifact hash, and import result.
6. Run static constructor/CLI checks without allocating model weights.
7. Re-verify the vendored manifest.

## Pass criteria

- **Unmodified artifact viable:** editable install provides the documented package and C++ extension; import and CLI config construction succeed without repository edits.
- **Partially viable:** extension or individual source components work, but packaging/configuration prevents documented startup.
- **Not viable:** extension cannot build/import or source has a deeper runtime incompatibility after dependencies are satisfied.

## Boundaries

- No full model load and no GPU timing in this protocol.
- No edits to `vendor/author-morphserve`.
- No claim of verified authorship.
- Missing packages may be installed from existing public/cached dependencies, but no gated model/data access is bypassed.
- A failure is preserved; retry only after a recorded changed hypothesis.

## Follow-up gate

Only after this protocol may a separate `runtime/` reconstruction normalize package names, CLI/config plumbing, and supported checkpoint loading. Mechanism semantics must remain unchanged until independent correctness tests expose a specific defect.

## Pre-registered follow-up after initial result

The initial extension import failed with `libc10.so: cannot open shared object file`, but the probe imported `swiftllm_c` before `torch`. PyTorch extensions commonly rely on PyTorch loading its shared libraries first. The changed hypothesis is that `import torch; import swiftllm_c` will succeed without rebuilding or changing source. Run exactly this additional probe once, preserve both import outcomes, and do not reinterpret the known packaging/config failures.

# Attempt 1 — setup failure

The pre-registered parity experiment did not reach model loading. An unrelated inherited `MODEL_PATH` overrode the runner's local snapshot default, and the resolved path did not exist. Dependencies and the reconstructed extension built; parity metrics were not produced and GPU experiment time was zero.

Evidence: `results/commands.txt`, `run.stderr.log`, `run.exitcode`, and `doc/debug-report-fp16-path.md`.

Changed hypothesis for one retry: use a collision-resistant `MORPHSERVE_MODEL_PATH` override and validate `config.json` before creating the environment.

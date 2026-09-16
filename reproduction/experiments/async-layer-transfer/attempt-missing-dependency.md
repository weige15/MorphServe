# Async seam rerun — missing dependency

After the memory-manager alignment runner rebuilt the shared virtual environment with only PyTorch, the async test runner failed during collection because it assumed `safetensors` remained installed. No CUDA test ran. `results-attempt-missing-dependency/metrics.json` is stale from the preceding successful run because the runner had not cleaned its output directory; the failed `test.log`/exit code take precedence.

Repair: make the runner recreate the locked base environment before building/running, so execution no longer depends on prior runner order.

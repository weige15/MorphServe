# Attempt 2 — eager optional dependency failure

The corrected local path was used and Transformers successfully loaded/executed the 8B reference. Candidate import then failed before loading its model because `swiftllm.worker.model` imports `GB/MB` from `swiftllm.utils`, whose module eagerly imports the optional evaluation package `evaluate`. The parity environment intentionally contains serving dependencies only.

This is an import-layer defect, not a numerical result. The changed retry will move `import evaluate` into `rouge_calculate`, its only caller, leaving metric behavior unchanged while allowing core serving imports.

Evidence: `results-attempt-2/run.stderr.log`, `commands.txt`, `run.exitcode`.

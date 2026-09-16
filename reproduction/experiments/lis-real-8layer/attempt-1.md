# Attempt 1 — generalized gate bug

All 36 real conditioned evaluations completed, all restores/final FP16 logits were exact, and the order `[25,24,26,27,28,29,30,31]` was produced. The command nevertheless exited 1 because `lis_real_pilot.py` retained three-layer gate constants (`6` calls, `3` steps) instead of deriving `n(n+1)/2` and `n` from the requested layer list.

This is a harness-verification failure after successful model work. The changed retry generalizes only the gate counts, then reruns the same frozen protocol so the final process and verifier exit 0.

# Bounded Marlin tuning follow-up

Status: **recorded after default-Marlin serving, before tuning measurements**.

Default vLLM 0.11.2 AWQ-Marlin removed KV pressure but missed FP16 at the
frozen scale-6 P95 by 0.075–0.285 s. Source inspection had already identified
two installed Marlin controls that do not change quantization, scheduler,
attention, KV semantics, or kernel family:

1. disable Marlin FP32 global reduction (`use_fp32_reduce=False`), which the
   installed source explicitly documents as reducing memory movement when
   performance is an issue;
2. additionally enable vLLM's existing `VLLM_MARLIN_USE_ATOMIC_ADD=1` path for
   small-N FP16 outputs on SM 8.6.

This is a bounded same-backend follow-up, not a load search. Evaluate exactly
the default, FP16-reduction, and FP16-reduction-plus-atomic settings on the
integrated 1024-token prefill/one-token decode microbenchmark (seven samples)
for W4-16. Rerun packed numerical sanity and a full-shape W4-16 profile for any
setting that improves median prefill by at least 3% without more than 3% decode
regression. Advance at most the best qualifying setting. If none qualifies,
retain the default result and stop tuning; do not try more flags or kernels.

If a setting advances, freeze it in run metadata and rerun the complete Phase C
matrix and the same pre-existing scale-6 workload/repeats. It must still beat
FP16 under the original eligibility rule; the scale or metric cannot change.
All default-backend raw evidence remains preserved.

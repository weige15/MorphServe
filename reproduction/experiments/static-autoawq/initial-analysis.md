# Initial static AutoAWQ result

The real packed W4 mechanism executed, but the confirmatory gate failed because two repeated logits were not bit-exact.

Positive evidence:

- 224 `WQLinear_GEMM` modules;
- INT32 qweight/qzeros and FP16 scales;
- decoder packed storage including metadata: 3,625,975,808 bytes;
- total unique state storage: 5,727,854,592 bytes;
- finite logits, correct shape, FP16/W4 top-1 both token 505.

Negative evidence:

- `repeat_logits_exact=false`, so the predeclared deterministic fixed-W4 gate failed;
- W4-vs-FP16 relative logit L2 was 0.35148 and max absolute difference 2.9414 on this prompt (quality observation, not a gate).

Source inspection after the run found the small-input AutoAWQ Triton path uses eight split-K partitions and `tl.atomic_add`. A locked follow-up will quantify five-repeat variance without relaxing the exact-repeat gate.

# Static AutoAWQ W4 baseline result

## Confirmatory verdict

**Failed the predeclared exact-repeat gate.** Five real packed W4 forwards were not bit-exact, so the experiment is classified **tested-not-reproduced** for deterministic fixed-W4 logits.

## Positive mechanism evidence

- 224 `awq.modules.linear.gemm.WQLinear_GEMM` modules executed.
- Actual packed storage: 3,489,660,928 qweight bytes + 27,262,976 qzeros bytes + 109,051,904 scale bytes = **3,625,975,808 decoder packed bytes including metadata**.
- Dtypes were INT32 qweight/qzeros and FP16 scales; this is not fake-quantized FP16.
- Total unique checkpoint state was 5,727,854,592 bytes.
- All five repeats were finite, shape-correct, and had stable top-1 token 505 and stable top-5 `[505,11,304,198,627]`.
- Against FP16, W4 retained top-1 but had relative logit L2 0.35126 and max absolute difference 2.9492 on this prompt.

## Repeat variance diagnosis

Relative differences from repeat 1 were `[0, 0.004930, 0.002256, 0.004203, 0.002260]`; maximum absolute differences were `[0, 0.058594, 0.039062, 0.058594, 0.035156]`. AutoAWQ 0.2.9's small-input Triton GEMM hard-codes eight split-K partitions and combines them with `tl.atomic_add`, which explains order-dependent FP accumulation. Top-k stability shows bounded behavior here, but exact equality is not defensible.

The paper does not claim bit-exact repeatability. This negative result affects the added verification practice and means later controlled-switch tests must use a tolerance fixed from independent W4 variance, not exact equality.

## Classification

- Real static packed W4 execution/storage: **approximate/modified-condition reproduced**.
- Bit-exact repeated W4 logits: **tested-not-reproduced**.
- Candidate in-place W4 adapter: still unimplemented; candidate's public llm-awq API/checkpoint format does not match this AutoAWQ asset.

## Evidence

- `results/metrics.json` and `verification.log`
- `results-initial/metrics.json`
- `scripts/static_autoawq_baseline.py`

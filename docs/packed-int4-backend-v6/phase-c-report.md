# Packed INT4 backend v6 — Phase C resource/performance gate

Status: **PASS — all three AWQ states advance to frozen serving**.

Every profile ran in a fresh process on RTX 3090 GPU 5 using the exact v5
full-shape configuration: 32 sequences, 49,152 tokens, block size 16, and 0.99
GPU utilization. Each state has two profiles. Isolated 1024-token prefill and
one-token decode use seven measured samples after warmup in the same
Torch-2.9/vLLM environment. Raw JSON and the derived table are under
`benchmark-results/packed-int4-backend-v6/phase-c/`.

| state | persistent GiB | packed GiB | temporary GiB | safe blocks (both reps) | slots | block gain | prefill median/P95 ms | decode median/P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 14.989 | 0 | 4.145 | 1768 / 1768 | 28,288 | reference | 260.78 / 266.80 | 22.18 / 23.77 |
| AWQ W4-8 | 12.585 | 0.844 | 4.145 | 3034 / 3034 | 48,544 | +71.6% | 263.54 / 267.82 | 22.15 / 22.38 |
| AWQ W4-16 | 10.181 | 1.689 | 4.145 | 4286 / 4286 | 68,576 | +142.4% | 264.54 / 271.24 | 22.25 / 23.02 |
| AWQ W4-32 | 5.374 | 3.377 | 4.142 | 6766 / 6766 | 108,256 | +282.7% | 263.91 / 281.00 | 21.30 / 21.94 |

Block and temporary-allocation coefficients of variation are 0% for every
state. Safe capacity is strictly monotonic. Unlike NF4 v5, every partial state
has substantially more safe KV than FP16: fused AWQ-Marlin keeps the same
~4.145-GiB full-shape activation/workspace envelope while persistent weights
fall by ~2.404 GiB per eight selected layers.

Median isolated prefill is only 1.1%, 1.4%, and 1.2% slower than FP16 for
W4-8/16/32. Decode is 0.1% faster, 0.3% slower, and 3.9% faster respectively.
Thus no candidate has a compute regression large enough to reject the measured
71.6–282.7% capacity relief before serving.

Because the capacity change is large, the final run plan is frozen to the full
v5 three-load matrix rather than near-knee alone: one run at scale 8, two at
the scale-6 FP16 knee, and one at scale 4 for FP16 and each AWQ state. This
selection is made before observing any AWQ serving result. Eligibility remains
decided only from the two matched scale-6 repeats and supporting telemetry;
adjacent loads explain the curve.

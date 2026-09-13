# Bounded AWQ QKV fusion follow-up

Status: **recorded after reduction flags failed, before fused-QKV measurement**.

The default backend misses scale-6 FP16 P95 by only 0.075–0.120 s at W4-8,
while packed MLP fusion already proved that concatenating compatible AutoAWQ
output columns before the same installed Marlin repack can remove launches and
intermediate buffers without changing semantics. Q, K, and V share input K and
have a valid combined Marlin N=6144.

Evaluate exactly one further same-backend optimization: packed `[q,k,v]`
fusion. First compare separate versus fused actual-checkpoint Marlin kernels at
M={1,1024,8192}, at least 20 samples after 10 warmups, including numerical
agreement with the three separate packed outputs. Implement it only if fused
QKV is numerically sane, has no additional persistent copy, improves the summed
kernel median at M=1024 or 8192 by at least 2%, and does not regress M=1 by more
than 3%.

If implemented, rerun the W4-8 integrated 1024-prefill/decode microbenchmark
and full-shape profile. Advance only if integrated prefill improves by at least
2% from the committed default W4-8 Phase C median (263.535 ms), decode regresses
by at most 3%, and workspace/safe blocks do not regress. If it advances, freeze
fused QKV, rerun packed sanity and all Phase C states, then rerun the same
scale-6 FP16/AWQ matrix. Otherwise stop backend tuning. No additional fusion,
flag, kernel, or load search is allowed.

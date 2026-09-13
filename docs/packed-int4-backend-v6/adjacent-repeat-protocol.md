# Exploratory high-load repeat protocol

Status: **recorded after the frozen 16-run matrix, before added repeats**.

The pre-registered scale-4 runs unexpectedly showed AWQ W4-16 with 2.343 s
P95 TTFT versus FP16 at 15.779 s, while W4-8 and W4-32 did not show the same
magnitude. This cannot change eligibility because every AWQ state already
failed both frozen scale-6 repeats. To determine whether the high-load
pressure-relief mechanism itself is real rather than a one-run artifact, add
exactly one scale-4 repeat for FP16 and AWQ W4-16, with the identical frozen
64-request workload and all original settings. Do not add or select a new load,
do not tune either backend, and report this only as exploratory adjacent-load
evidence.

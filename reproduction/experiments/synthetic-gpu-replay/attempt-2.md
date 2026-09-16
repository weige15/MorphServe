# Attempt 2 — transient GPU contention

The corrected warmup run did not reach model execution. Physical GPU 1 had acquired a 10.32-GiB external process after the prior run; only 12.95 GiB remained, below the candidate's 14.96-GiB contiguous model allocation, causing a PyTorch OOM. Subsequent `nvidia-smi` showed every GPU materially occupied.

This is a recorded transient resource failure, not a code hypothesis failure. Retry the unchanged frozen protocol only when one GPU again has sufficient headroom; do not evict other users or reduce the model silently.

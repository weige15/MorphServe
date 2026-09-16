# Public trace audit and exact-window blocker

Primary public files were recovered and hashed in `results/raw/public-trace-audit.json`.

| Source | Version/hash | Requests | Duration | First 72 s | Densest 72 s (source characterization only) |
|---|---|---:|---:|---:|---:|
| Azure Code | introduced by `7909210...`; SHA `54e9a6...` | 8,819 | 3,435.95 s | 63 | 827 at relative 854.081 s |
| Azure Conversation | same release; SHA `2f1e5b...` | 19,366 | 3,501.72 s | 249 | 616 at relative 1,634.755 s |
| BurstGPT_1 | release v1.1 (2024-06-13); SHA `4bb378...` | 1,429,737 | 5,269,968 s | 2 | 1,666 at timestamp 4,450,025 s |

BurstGPT v1.1 is the defensible release available before the MorphServe paper; current v2 was published in 2026 and must not silently replace it.

Exact-condition reproduction remains blocked because neither paper nor public candidate history identifies:

1. Azure Code versus Conversation;
2. either 72-second start offset;
3. whether 4.75×/1.75× means multiplying or dividing inter-arrivals, thinning, or stochastic downsampling;
4. sampled task examples, mapping order, or seed.

The first and densest windows differ by over an order of magnitude. Choosing either after observing serving outcomes would materially cherry-pick load. Densest-window calculations are therefore source characterization, not selected evaluation traces.

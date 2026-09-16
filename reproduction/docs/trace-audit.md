# Public trace audit and exact-window blocker

Primary public files were recovered and hashed in `results/raw/public-trace-audit.json`.

| Source | Version/hash | Requests | Duration | First 72 s | Densest 72 s (source characterization only) |
|---|---|---:|---:|---:|---:|
| Azure Code | introduced by `7909210...`; SHA `54e9a6...` | 8,819 | 3,435.95 s | 63 | 827 at relative 854.081 s |
| Azure Conversation | same release; SHA `2f1e5b...` | 19,366 | 3,501.72 s | 249 | 616 at relative 1,634.755 s |
| BurstGPT_1 | release v1.1 (2024-06-13); SHA `4bb378...` | 1,429,737 | 5,269,968 s | 2 | 1,666 at timestamp 4,450,025 s |

BurstGPT v1.1 is the defensible release available before the MorphServe paper; current v2 was published in 2026 and must not silently replace it.

## Figure-derived window inference

A later source-only comparison digitized the approximate per-second token-volume shapes in PDF Figure 1b and ranked every dense 73-bin source segment before any serving outcomes (`configs/figure1b-digitized.json`, `scripts/infer_trace_windows_from_figure.py`). Both request-count and source-token rankings select the same candidate for each trace:

- **Azure Code**, relative second **1073** (`2023-11-16 18:34:56.979960`), 443 raw requests in `[1073,1145)`. Context-volume Pearson correlation is 0.8793 versus 0.5706 for the runner-up.
- **BurstGPT v1.1**, timestamp second **1,781,278**, 214 raw requests in `[1781278,1781350)`. Request-token Pearson correlation is 0.7356 versus 0.6173 for the runner-up among dense windows.

Figure 1a independently matches Azure Code's zero intervals and ≈1.4M-token/min maximum, not the continuously busy Conversation file. Section 5 explicitly says evaluation uses the representative 72-second snippets in Figure 1, so these are strong **figure-inferred evaluation-window candidates**, frozen in `configs/figure1b-inferred-trace-windows.json`. They remain approximate rather than explicit author-published offsets: digitization/integer binning cannot recover a sub-second boundary, and task-context replacement changes token volume.

Exact-condition reproduction therefore remains blocked by:

1. the exact sub-second boundaries/plot aggregation convention;
2. whether 4.75×/1.75× means timestamp scaling, deterministic/stochastic thinning, or another operation and its seed;
3. sampled task examples, mapping order, or seed.

The first and densest windows differ by over an order of magnitude. Choosing either after observing serving outcomes would materially cherry-pick load. They remain source characterization, while only the figure-inferred candidates may be used in predeclared modified-condition replays.

For one explicit reconstruction, `configs/modified-trace-manifest.json` freezes systematic index thinning (retain row `floor(k×factor)`, preserve timestamps) before outcomes. It yields 94 Azure and 123 Burst requests under `traces/figure1b-inferred/`. This operation is deterministic and auditable but **not recovered author behavior**, and contexts remain intentionally unmapped.

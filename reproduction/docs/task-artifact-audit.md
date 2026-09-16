# Task and quality-artifact audit

Captured primary-source metadata is immutable under `sources/task-source-audit/` with `MANIFEST.sha256`.

| Task | Paper-linked/public source | Pinned source state | What is recoverable | Exact-condition gap |
|---|---|---|---|---|
| GovReport | `launch/gov_report` on Hugging Face | `32feeaede49fed993aef070bc4da09263fd0429a` (2022-11-09) | CRS/GAO train/valid/test files and loader are public. LongBench alternative is `THUDM/LongBench@5e628be...` (2024-12-18). | Paper does not name CRS vs GAO, split, rows, prompt template, sampling seed, or whether main Figure 4 uses LongBench preprocessing. |
| QMSum | `Yale-LILY/QMSum` | `83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e` | Raw task repository is public and MIT licensed. | No release/tag, exact split/meeting/query rows/prompt/reference formatting/sample order absent. |
| DuReader | `baidu/DuReader` | `c625076b06da8f56d59f19c41c73bd580a98a347` | Chinese upstream code/data instructions are public. | Recursive 303-path tree has no English/translation artifact and repository has no releases. The paper's statement that the English-translated version is “hosted” at this URL is not supported by the linked tree. Translation system/version, translated rows, split and normalization are absent. |
| Multi-News | `Alex-Fabbri/Multi-News` | `50615eae2d20c44666197a9c76a1b4317afc382d` | Raw task repository is public. | No release/tag; exact split/rows/prompt/reference formatting/sample order absent. |
| BookSum Chapters | `salesforce/booksum` | defensible pre-paper commit `9ef00dc962c40784cc4717936a3f7d37028e6268` (2022-02-24); current HEAD is post-paper | Chapter task source is public. | Exact chapter examples, 6K truncation/padding/prompt, 2K decoding/EOS and metric preprocessing absent. |

## Candidate-code evidence

The candidate snapshot contains only generic `tokenize_long_bench(dataset, tokenizer, prompt_max_len, reference_max_len)`. It requires already-created `prompt` and `reference` columns, left-pads prompts and pads references to fixed maxima, but does not create the task prompts, choose rows/splits, translate DuReader, set a sampling seed, or record metric package versions. No experiment script/config/raw generation survives in candidate history.

## Consequence

Public upstream data can support explicitly modified-condition quality pilots, but cannot recover the paper's exact request-to-context mapping or numerical quality tables. In particular, substituting Chinese DuReader, LongBench rows, or a new machine translation would be a material deviation and must not be used for an exact Table 2 claim.

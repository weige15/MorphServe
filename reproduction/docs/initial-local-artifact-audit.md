# Research: MorphServe MLSys 2026 author-artifact audit (limited local evidence)

## Summary

The available checkout does **not** contain an implementation, traces, translated DuReader data, experiment configs, or machine-readable results: its Git index contains only `README.md` and `references/morphserve-2506.02006-v2.pdf`, and the README is only `# MorphServe`. It does contain an arXiv v2 paper PDF and that PDF identifies arXiv `2506.02006v2`, the authors, an Azure trace URL, and upstream SwiftLLM (`interestingLSY/swiftLLM`).

This is **not an attestation of current remote state**. This worker had no network, shell, GitHub API, or web-search tool, so branches/tags/releases/issues/history at `ds2-lab/MorphServe`, author pages, arXiv’s live record, linked datasets, and upstream SwiftLLM could not be queried. The remote configured in the supplied checkout is also `weige15/MorphServe`, not the requested `ds2-lab/MorphServe`; remote conclusions would therefore be unsafe.

## Scope and evidence boundary

- **Evidence** means bytes read from the supplied checkout’s working tree or `.git` metadata.
- **Inference** is explicitly labeled and must not be treated as a live-public-availability finding.
- Local repo was not edited. The only written file is this required report outside the repo.
- No commands were available or run; inspection used read-only file access.

## Findings

1. **BLOCKER — no reproducible artifact is present in the supplied tree.** `README.md` contains exactly one line, `# MorphServe`. The Git index has exactly two entries: `README.md` and `references/morphserve-2506.02006-v2.pdf`. Thus the local snapshot has no source tree, install/environment file, scripts, trace files, translated DuReader files, configs, checkpoints, result tables, or plotting code. Evidence: local paths `README.md`, `.git/index`.

2. **HIGH — repository identity mismatch.** `.git/config` sets `remote.origin.url = git@github.com:weige15/MorphServe.git`, while the audit target is `https://github.com/ds2-lab/MorphServe`. The fetch refspec is `+refs/heads/*:refs/remotes/origin/*`. This may be a private/personal staging remote or a different repository; without network evidence, equivalence cannot be inferred. Evidence: local `.git/config`. URLs: [requested repository](https://github.com/ds2-lab/MorphServe), [configured repository](https://github.com/weige15/MorphServe).

3. **HIGH — local Git history is minimal and potentially post-dated.** Local `HEAD`, `refs/heads/main`, and `refs/remotes/origin/main` all resolve to `6488af573cc9e6f2f26272cd8231c4d05cc014eb`; `.git/FETCH_HEAD` records the same object as branch `main` from `github.com:weige15/MorphServe`. Reflogs show an initial object `dba085ad0cf14fbf6f75eaee33a972eb43eb49dc` and a later commit `6488af573cc9e6f2f26272cd8231c4d05cc014eb` with message `added reference`. Raw reflog times are `1789570282 +0800` (initial commit/rename) and `1789571760 +0800` (second commit), with the remote-tracking update at `1789571774 +0800`. These raw Unix timestamps appear future-dated relative to the PDF metadata and should be checked against the live GitHub commit `committer.date`; no chronology claim is made here. Evidence: `.git/HEAD`, `.git/refs/heads/main`, `.git/refs/remotes/origin/main`, `.git/FETCH_HEAD`, `.git/logs/HEAD`, `.git/logs/refs/heads/main`, `.git/logs/refs/remotes/origin/main`.

4. **MEDIUM — no local submodule or Git LFS declaration.** `.gitmodules` and `.gitattributes` are absent at this checkout. Therefore the two indexed files are not accompanied by a declared submodule mapping or tracked LFS patterns in this snapshot. This does **not** prove that no other remote ref/history contains submodules or LFS pointers. Evidence: missing local paths `.gitmodules`, `.gitattributes`; two-entry `.git/index`.

5. **VERIFIED PAPER ARTIFACT — arXiv v2 PDF is locally present.** `references/morphserve-2506.02006-v2.pdf` is a 19-page PDF. Embedded metadata gives title “MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing”; authors Zhaoyuan Su, Zeyu Zhang, Tingfeng Lan, Zirui Wang, Haiying Shen, Juncheng Yang, Yue Cheng; DOI `10.48550/arXiv.2506.02006`; arXiv ID `https://arxiv.org/abs/2506.02006v2`; CC BY 4.0; and XMP metadata time `2026-01-08T01:19:37.768828+00:00`. URLs: [arXiv v2](https://arxiv.org/abs/2506.02006v2), [versioned PDF](https://arxiv.org/pdf/2506.02006v2), [DOI](https://doi.org/10.48550/arXiv.2506.02006).

6. **VERIFIED LINK PROVENANCE — upstream SwiftLLM is cited, but no pin/patch is locally supplied.** Embedded PDF links point to `https://github.com/interestingLSY/swiftLLM`; the PDF name tree includes citation keys `jiang2024neo_swiftLLM` and `swiftllm_github_repo`. No SwiftLLM submodule, vendored tree, patch, commit hash, lockfile, or setup instructions exist in the local checkout. Therefore exact upstream provenance is unresolved. [SwiftLLM upstream](https://github.com/interestingLSY/swiftLLM).

7. **VERIFIED LINK — the paper points to Microsoft’s Azure LLM inference dataset page, but no trace snapshot is local.** The PDF embeds `https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md`. No trace file, checksum, download script, selected workload identifier, preprocessing script, arrival-rate transform, or random seed exists locally. [Azure LLM inference dataset page](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md).

8. **HIGH — translated DuReader availability is unverified and absent locally.** The PDF has a DuReader citation (`cite.he2017dureader`) but this checkout contains neither translated data nor a translation/download/preprocessing script. The absence is evidence only for this local snapshot. Whether a translated DuReader artifact is currently linked from GitHub, an author page, Hugging Face, or another branch cannot be answered without live checks.

9. **HIGH — exact reported results cannot be regenerated from this snapshot.** The PDF is evidence that results were published, not that raw measurements are available. There are no executable configs, manifests, run logs, result CSV/JSON files, figure-generation sources, or checksums in the indexed tree. Exact numerical reproduction therefore remains blocked even if the public upstream datasets and models are independently obtainable.

## Actionable live-remote audit checklist

Run these against both `ds2-lab/MorphServe` and `weige15/MorphServe`; record HTTP status, response timestamp, immutable SHA/asset ID, `created_at`, `updated_at`, `published_at`, and `pushed_at`. Use `Accept: application/vnd.github+json` and GitHub API version `2022-11-28`.

```bash
# Repository identity/default branch/timestamps/license/archive state
curl -sS -D headers.txt -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2022-11-28' \
  https://api.github.com/repos/ds2-lab/MorphServe

# Every branch/tag/ref/release/commit (paginate Link headers)
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/branches?per_page=100'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/tags?per_page=100'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/git/matching-refs/heads/'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/git/matching-refs/tags/'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/releases?per_page=100'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/commits?per_page=100'

# For EACH branch/tag SHA: complete recursive tree; mode 160000 denotes submodules
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/git/trees/<SHA>?recursive=1'

# Issues and PRs (the issues endpoint includes PRs); then timelines/comments
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/issues?state=all&per_page=100'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/pulls?state=all&per_page=100'
curl -sS -H 'Accept: application/vnd.github+json' \
  'https://api.github.com/repos/ds2-lab/MorphServe/issues/<N>/timeline?per_page=100'

# Repository-linked artifacts and history surfaces
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/actions/artifacts?per_page=100'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/deployments?per_page=100'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/forks?per_page=100&sort=newest'
curl -sS 'https://api.github.com/repos/ds2-lab/MorphServe/contributors?per_page=100&anon=1'

# Clone all refs without altering the supplied checkout
mkdir /tmp/morphserve-audit && cd /tmp/morphserve-audit
git clone --mirror https://github.com/ds2-lab/MorphServe.git repo.git
cd repo.git
git show-ref
git log --all --date=iso-strict --decorate --stat --oneline
git fsck --full --no-reflogs --unreachable

# Enumerate every path/ref and identify LFS pointers/submodules
for r in $(git for-each-ref --format='%(refname)'); do
  git ls-tree -rl "$r"
done
git grep -n 'version https://git-lfs.github.com/spec/v1' $(git rev-list --all)
git log --all -- .gitattributes .gitmodules
```

For LFS, inspect `.gitattributes`, pointer OIDs/sizes, and test the authenticated batch endpoint (POST body uses discovered OIDs):

```text
POST https://github.com/ds2-lab/MorphServe.git/info/lfs/objects/batch
Content-Type: application/vnd.git-lfs+json
Accept: application/vnd.git-lfs+json
```

ArXiv checks:

```bash
curl -sS https://export.arxiv.org/api/query?id_list=2506.02006
curl -sS -I https://arxiv.org/abs/2506.02006v1
curl -sS -I https://arxiv.org/abs/2506.02006v2
curl -sS -I https://arxiv.org/pdf/2506.02006v2
curl -sS -I https://arxiv.org/e-print/2506.02006v2
```

SwiftLLM provenance checks:

```bash
curl -sS https://api.github.com/repos/interestingLSY/swiftLLM
curl -sS 'https://api.github.com/repos/interestingLSY/swiftLLM/commits?per_page=100'
curl -sS 'https://api.github.com/repos/interestingLSY/swiftLLM/tags?per_page=100'
# Compare MorphServe code/history for copied paths, notices, commit IDs, and patches.
```

Also inspect commit diffs and all issue/PR bodies/comments for URLs; GitHub code search requires authentication and should query `repo:ds2-lab/MorphServe (DuReader OR trace OR config OR SwiftLLM OR huggingface OR dataset)`. Search each author’s institutional/personal page and Hugging Face organization/user pages, preserving exact URL and observed timestamp. These checks were **not run** here.

## Smallest missing inputs for exact reproduction

1. **Immutable implementation:** public source at a named commit SHA, including any CUDA/C++ kernels and complete MorphServe-vs-SwiftLLM diff; exact upstream SwiftLLM SHA.
2. **Executable environment:** container digest or lockfiles plus CUDA, driver, PyTorch, compiler, GPU architecture, and build flags.
3. **Model provenance:** exact model repository revisions, tokenizer revisions, weight checksums, quantization/calibration procedure, calibration samples/seeds, and any derived quantized weights.
4. **Workloads:** exact Azure trace files and checksums, selected intervals, preprocessing/scaling code and parameters; translated DuReader file/checksum, source split/version, translation method or script/model revision, prompt formatting, and redistribution/license statement; equivalent exact inputs for every other workload.
5. **Experiment configs:** one versioned config/command per paper table/figure, including baselines, memory budgets, SLOs, request-rate multipliers, sequence-length limits, layer precision/swap policy, KV resize policy, warmup, repetitions, and random seeds.
6. **Results provenance:** raw per-request/per-run outputs, logs, aggregation/statistics scripts, and plotting/table scripts with expected hashes.
7. **Hardware protocol:** exact GPU/CPU/storage topology, GPU count, clocks/power mode, NVLink/PCIe details, storage/GDS setup, concurrency, and isolation procedure.

Items 1–5 are the minimum to attempt exact execution; item 6 is the minimum to verify that regenerated numbers match the publication rather than merely running a similar experiment.

## Sources

- **Kept:** local `README.md` — direct evidence of the checkout’s documentation content.
- **Kept:** local `.git/index` — direct evidence of the two tracked paths.
- **Kept:** local `.git/config`, refs, FETCH_HEAD, and reflogs — direct evidence of configured origin, local SHAs, and raw local timestamps.
- **Kept:** local `references/morphserve-2506.02006-v2.pdf` — primary paper artifact and embedded metadata/URLs.
- **Kept:** [arXiv 2506.02006v2](https://arxiv.org/abs/2506.02006v2) — URL embedded in the local PDF; live page not fetched.
- **Kept:** [SwiftLLM](https://github.com/interestingLSY/swiftLLM) — upstream URL embedded in the local PDF; live repository not fetched.
- **Kept:** [Azure LLM inference dataset](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) — dataset URL embedded in the local PDF; live page not fetched.
- **Dropped/unverified:** live `ds2-lab/MorphServe` GitHub pages/API, releases, issues, PRs, branches, tags, author pages, arXiv API/version history, Hugging Face artifacts, and live SwiftLLM history — unavailable because this worker had no network/search/shell tool.

## Gaps and residual risks

- **Primary gap:** current public availability “as of now” cannot be established from this execution. A parent/reviewer with network access must run the checklist and timestamp the responses.
- Loose refs beyond the directly readable known paths could not be directory-listed; absence of `.git/packed-refs` is not a complete proof that no other loose refs exist locally.
- The PDF’s compressed page text was not extracted by a PDF tool; findings use embedded metadata, names, and URI annotations. Exact experimental prose should be checked in the rendered/source paper.
- GitHub may expose artifacts through issue comments, release assets, Actions artifacts, forks, commit history, Git LFS, or external links even when the default branch is sparse.
- Author-owned artifacts can move or be silently replaced; preserve response headers, hashes, release asset IDs, and content checksums.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Concrete severity-ranked findings identify README.md, references/morphserve-2506.02006-v2.pdf, .git/config, refs, FETCH_HEAD, reflogs, exact local SHAs/timestamps, embedded primary-source URLs, and residual risks."
    }
  ],
  "changedFiles": [
    "/nfs/home/s314511048/.pi/agent/sessions/--nfs-home-s314511048-MorphServe--/subagent-artifacts/outputs/1bfd2235-29ea-4b1c-a14a-93c8fa047bd9/research.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "No shell commands available; read-only file inspection via runtime file API",
      "result": "not-run",
      "summary": "README, Git metadata/index, and local PDF were inspected; no live network validation was possible."
    }
  ],
  "validationOutput": [
    "Local Git index exposes exactly README.md and references/morphserve-2506.02006-v2.pdf.",
    "HEAD/main/origin-main locally resolve to 6488af573cc9e6f2f26272cd8231c4d05cc014eb.",
    "Configured origin is git@github.com:weige15/MorphServe.git, not ds2-lab/MorphServe.",
    "PDF metadata identifies arXiv 2506.02006v2 and embeds SwiftLLM and Azure trace URLs."
  ],
  "residualRisks": [
    "Current remote state, all branches/tags/releases/issues/LFS/submodules, author pages, and external artifacts remain unverified because no network/search/shell tool was available.",
    "The local checkout targets a different GitHub owner than the requested public repository.",
    "Loose refs could not be directory-enumerated, and compressed PDF body text was not fully extracted."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added only the required external research report; did not modify the local MorphServe repository.",
  "reviewFindings": [
    "blocker: README.md:1 - contains only '# MorphServe'; no build/run/reproduction instructions.",
    "blocker: repository tree - only README.md and references/morphserve-2506.02006-v2.pdf are indexed; implementation, traces, translated DuReader, configs, and raw results are absent locally.",
    "high: .git/config - origin is weige15/MorphServe rather than the requested ds2-lab/MorphServe, so repository identity/provenance must be resolved.",
    "high: SwiftLLM provenance - paper links interestingLSY/swiftLLM but the checkout provides no upstream commit pin or patch.",
    "high: workload provenance - paper links Azure's dataset and cites DuReader, but no exact trace snapshot, preprocessing, translated DuReader artifact, checksums, or seeds are local."
  ],
  "manualNotes": "Strictly limited local-evidence audit authorized by supervisor. The live URL/API checklist must be run by a network-enabled reviewer before making any as-of-now public-availability claim."
}
```

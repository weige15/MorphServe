"""Prepare a fixed, auditable static-quantization workload from public sources.

The script deliberately chooses the first 72-second half-open BurstGPT window
with at least 20 rows, applies the paper's 1.75x time downscaling, and maps
those rows to the public DuReader demo test examples in order.  It writes both
an exact request manifest and the selected trace/data records so benchmark
runs never need to reread a mutable external dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SEGMENT_SECONDS = 72
TRACE_SCALE = 1.75
PROMPT_TOKENS = 1024
RESPONSE_TOKENS = 512
MIN_SEGMENT_ROWS = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"trace is missing columns: {sorted(missing)}")
    rows.sort(key=lambda row: (int(row["Timestamp"]),))
    return rows


def choose_segment(rows: list[dict[str, str]]) -> tuple[int, list[dict[str, str]]]:
    timestamps = [int(row["Timestamp"]) for row in rows]
    for start_index, start in enumerate(timestamps):
        end_index = start_index
        while end_index < len(rows) and int(rows[end_index]["Timestamp"]) < start + SEGMENT_SECONDS:
            end_index += 1
        if end_index - start_index >= MIN_SEGMENT_ROWS:
            return start, rows[start_index:end_index]
    raise RuntimeError("no 72-second BurstGPT window meets the minimum row count")


def load_dureader(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row.get("answers")]
    if len(rows) < MIN_SEGMENT_ROWS:
        raise ValueError(f"DuReader file has {len(rows)} answer-bearing rows; need at least {MIN_SEGMENT_ROWS}")
    return rows


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, return_attention_mask=False)["input_ids"])


def make_prompt(tokenizer: Any, record: dict[str, Any]) -> tuple[str, list[int]]:
    question = str(record.get("question", ""))
    documents = record.get("documents", [])
    paragraphs: list[str] = []
    for document in documents:
        title = str(document.get("title", ""))
        selected = document.get("paragraphs", [])
        if title:
            paragraphs.append(title)
        paragraphs.extend(str(paragraph) for paragraph in selected)
    context = "\n".join(paragraphs)
    prefix = (
        "请阅读以下材料并回答问题。只输出简洁答案，不要解释。\n"
        f"问题：{question}\n材料："
    )
    suffix = "\n答案："
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    # Preserve both the question/instruction and the answer cue.  The previous
    # prefix-only truncation could discard the cue for long documents, making
    # the quality measurement a different task from DuReader QA.
    padding_unit = tokenizer.decode(
        tokenizer.encode(" hello", add_special_tokens=False),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    # Search near the token-budget estimate rather than from the full document
    # length.  Some DuReader documents contain tens of thousands of tokens, and
    # repeatedly decoding every possible prefix would make preparation needlessly
    # expensive.  Boundary merges are handled by the small downward window.
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    estimated_keep = min(
        len(context_ids),
        max(0, PROMPT_TOKENS - 1 - len(prefix_ids) - len(suffix_ids) + 8),
    )
    lower_keep = max(0, estimated_keep - 64)
    for keep in range(estimated_keep, lower_keep - 1, -1):
        candidate_context = tokenizer.decode(
            context_ids[:keep],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prompt = prefix + candidate_context + suffix
        while token_count(tokenizer, prompt) < PROMPT_TOKENS:
            candidate = prompt[: -len(suffix)] + padding_unit + suffix
            if token_count(tokenizer, candidate) > PROMPT_TOKENS:
                break
            prompt = candidate
        if token_count(tokenizer, prompt) == PROMPT_TOKENS:
            return prompt, tokenizer(prompt, return_attention_mask=False)["input_ids"]
    raise ValueError(f"could not construct a prompt for question_id={record.get('question_id')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--dureader", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trace_rows = load_trace(args.trace)
    segment_start, segment_rows = choose_segment(trace_rows)
    dataset_rows = load_dureader(args.dureader)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer_sha256 = sha256_file(args.tokenizer / "tokenizer.json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_trace_path = args.output_dir / "burstgpt_segment.csv"
    with selected_trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=segment_rows[0].keys())
        writer.writeheader()
        writer.writerows(segment_rows)

    workload_path = args.output_dir / "workload.jsonl"
    workload_rows: list[dict[str, Any]] = []
    for sequence, trace_row in enumerate(segment_rows):
        dataset_row = dataset_rows[sequence]
        prompt, prompt_ids = make_prompt(tokenizer, dataset_row)
        answers = [str(answer) for answer in dataset_row.get("answers", [])]
        workload_rows.append(
            {
                "schema_version": 1,
                "request_id": f"table5-r{sequence:04d}",
                "sequence": sequence,
                "source_trace_timestamp_s": int(trace_row["Timestamp"]),
                "planned_arrival_offset_s": (int(trace_row["Timestamp"]) - segment_start) * TRACE_SCALE,
                "trace_model": trace_row["Model"],
                "trace_log_type": trace_row["Log Type"],
                "trace_request_tokens": int(trace_row["Request tokens"]),
                "trace_response_tokens": int(trace_row["Response tokens"]),
                "trace_total_tokens": int(trace_row["Total tokens"]),
                "dataset_question_id": dataset_row.get("question_id"),
                "dataset_question_type": dataset_row.get("question_type"),
                "dataset_reference_answers": answers,
                "prompt": prompt,
                "prompt_token_count": len(prompt_ids),
                "prompt_token_ids_sha256": hashlib.sha256(
                    json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "requested_output_token_count": RESPONSE_TOKENS,
            }
        )
    with workload_path.open("w", encoding="utf-8") as handle:
        for row in workload_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metadata = {
        "schema_version": 1,
        "segment_rule": f"first request-timestamp-started half-open 72-second window with at least {MIN_SEGMENT_ROWS} rows",
        "segment_start_timestamp_s": segment_start,
        "segment_end_timestamp_exclusive_s": segment_start + SEGMENT_SECONDS,
        "segment_row_count": len(segment_rows),
        "trace_scale": TRACE_SCALE,
        "scaled_offset_formula": "(Timestamp - segment_start_timestamp_s) * 1.75",
        "scaled_last_offset_s": workload_rows[-1]["planned_arrival_offset_s"],
        "prompt_token_count": PROMPT_TOKENS,
        "requested_output_token_count": RESPONSE_TOKENS,
        "trace_source": str(args.trace),
        "trace_sha256": sha256_file(args.trace),
        "selected_trace_sha256": sha256_file(selected_trace_path),
        "dureader_source": str(args.dureader),
        "dureader_sha256": sha256_file(args.dureader),
        "tokenizer_source": str(args.tokenizer),
        "tokenizer_json_sha256": tokenizer_sha256,
        "workload_sha256": sha256_file(workload_path),
        "language_note": "official public DuReader 2.0 Chinese demo; paper's English-translated subset unavailable",
        "mapping_rule": "BurstGPT segment row i -> i-th answer-bearing DuReader demo row",
        "request_ids": [row["request_id"] for row in workload_rows],
    }
    (args.output_dir / "workload_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

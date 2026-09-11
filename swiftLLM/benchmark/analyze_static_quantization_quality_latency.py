"""Regenerate static quantization quality/latency aggregates and the Pareto plot from raw runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .metrics import derive_request_metrics, distribution, percentile


PAPER_TARGETS = {
    "fp16_0": {"f1_percent": 25.19, "ttft_p95_s": 5.65, "slo_violation_percent": 12.7},
    "w4_8": {"f1_percent": 24.16, "ttft_p95_s": 2.61, "slo_violation_percent": 4.2},
    "w4_16": {"f1_percent": 23.93, "ttft_p95_s": 1.74, "slo_violation_percent": 0.0},
    "w4_32": {"f1_percent": 23.66, "ttft_p95_s": 1.62, "slo_violation_percent": 0.0},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_chars(text: str) -> list[str]:
    return [char for char in text if len(char.strip()) != 0]


def local_f1(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_reference_f1(answer: str, references: list[str]) -> float:
    pred = normalize_chars(answer or "")
    return max((local_f1(pred, normalize_chars(ref)) for ref in references), default=0.0)


def analyze_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    requests = read_jsonl(run_dir / "requests.jsonl")
    telemetry = read_jsonl(run_dir / "telemetry.jsonl")
    quality_rows: list[dict[str, Any]] = []
    for row in requests:
        metrics = derive_request_metrics(row)
        refs = [str(value) for value in row.get("dataset_reference_answers", [])]
        answer = row.get("generated_answer") or ""
        f1 = best_reference_f1(answer, refs) if row.get("status") == "completed" else None
        ttft = metrics.get("ttft_s")
        slo_violation = bool(isinstance(ttft, (int, float)) and ttft > 2.0)
        quality_rows.append({
            "condition": metadata.get("condition"),
            "quantized_layer_count": metadata.get("quantized_layer_count"),
            "run_id": metadata.get("run_id"),
            "request_id": row.get("benchmark_request_id"),
            "sequence": row.get("sequence"),
            "status": row.get("status"),
            "dataset_question_id": row.get("dataset_question_id"),
            "reference_count": len(refs),
            "generated_answer": answer,
            "f1": f1,
            "ttft_s": ttft,
            "slo_violation": slo_violation if ttft is not None else None,
            "error": row.get("error"),
        })
    quality_path = run_dir / "quality_metrics.jsonl"
    with quality_path.open("w", encoding="utf-8") as handle:
        for row in quality_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    completed = [row for row in quality_rows if row["status"] == "completed"]
    f1_values = [float(row["f1"]) for row in completed if row["f1"] is not None]
    ttft_values = [float(row["ttft_s"]) for row in completed if row["ttft_s"] is not None]
    violations = [row for row in completed if row["slo_violation"]]
    launched = len(requests)
    condition = str(metadata.get("condition"))
    summary = {
        "run_id": metadata.get("run_id"),
        "condition": condition,
        "quantized_layer_count": metadata.get("quantized_layer_count"),
        "quantization_label": metadata.get("quantization_label"),
        "request_count_launched": launched,
        "completed_request_count": len(completed),
        "failed_or_incomplete_request_count": launched - len(completed),
        "quality_metric": "macro mean of per-request character-overlap F1 against best reference",
        "f1_percent": (sum(f1_values) / len(f1_values) * 100) if f1_values else None,
        "f1_distribution": distribution(f1_values),
        "ttft_s": distribution(ttft_values),
        "ttft_p95_s": percentile(ttft_values, 95),
        "slo_threshold_s": 2.0,
        "slo_violation_count": len(violations),
        "slo_violation_percent_of_launched": (len(violations) / launched * 100) if launched else None,
        "slo_denominator_policy": "all launched requests; failures/incomplete reported separately",
        "raw_telemetry_sample_count": len(telemetry),
        "peak_waiting_q_depth": max((row.get("waiting_q_depth", 0) for row in telemetry), default=None),
        "peak_logical_kv_utilization": max((row.get("logical_kv_utilization", 0) for row in telemetry), default=None),
        "num_gpu_blocks_observed": sorted({row.get("num_gpu_blocks") for row in telemetry}),
        "source_raw_files": ["metadata.json", "requests.jsonl", "telemetry.jsonl"],
        "quality_source": "requests.jsonl generated_answer and dataset_reference_answers",
        "paper_reference_target": PAPER_TARGETS.get(condition),
    }
    (run_dir / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def plot_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=180)
    colors = {"fp16_0": "#1f77b4", "w4_8": "#ff7f0e", "w4_16": "#2ca02c", "w4_32": "#d62728"}
    for row in rows:
        x, y = row.get("ttft_p95_s"), row.get("f1_percent")
        if x is None or y is None:
            continue
        condition = str(row["condition"])
        marker = "x" if "repeat" in str(row["run_id"]) else "o"
        ax.scatter(x, y, s=70, marker=marker, color=colors.get(condition, "black"), label=f"{condition} ({row['run_id']})")
        ax.annotate(condition, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    # Reference values are shown only as hollow gray crosses; they are not used
    # to modify or filter measured values.
    for condition, target in PAPER_TARGETS.items():
        ax.scatter(target["ttft_p95_s"], target["f1_percent"], marker="+", s=90, color="0.45", linewidths=1.5)
    ax.axvline(2.0, color="0.45", linestyle="--", linewidth=1, label="2 s SLO")
    ax.set_xlabel("P95 TTFT (s), measured")
    ax.set_ylabel("F1 (%) — measured macro mean")
    ax.set_title("Static Table-5 quality–latency trade-off")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [analyze_run(path, args.output_dir) for path in args.runs]
    summaries.sort(key=lambda row: (int(row.get("quantized_layer_count") or 0), str(row.get("run_id"))))

    results_path = args.output_dir / "static_quantization_quality_latency_results.csv"
    columns = ["condition", "quantized_layer_count", "run_id", "f1_percent", "ttft_p95_s", "slo_violation_percent_of_launched", "request_count_launched", "completed_request_count", "failed_or_incomplete_request_count", "num_gpu_blocks_observed", "quantization_label"]
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in columns})
    grouped: list[dict[str, Any]] = []
    for condition in ("fp16_0", "w4_8", "w4_16", "w4_32"):
        members = [row for row in summaries if row.get("condition") == condition]
        if not members:
            continue
        def values(key: str) -> list[float]:
            return [float(row[key]) for row in members if row.get(key) is not None]
        f1s = values("f1_percent")
        ttfts = values("ttft_p95_s")
        slos = values("slo_violation_percent_of_launched")
        grouped.append({
            "condition": condition,
            "quantized_layer_count": members[0].get("quantized_layer_count"),
            "run_count": len(members),
            "run_ids": [row.get("run_id") for row in members],
            "f1_percent_mean": sum(f1s) / len(f1s) if f1s else None,
            "f1_percent_min": min(f1s) if f1s else None,
            "f1_percent_max": max(f1s) if f1s else None,
            "ttft_p95_s_mean": sum(ttfts) / len(ttfts) if ttfts else None,
            "ttft_p95_s_min": min(ttfts) if ttfts else None,
            "ttft_p95_s_max": max(ttfts) if ttfts else None,
            "slo_violation_percent_mean": sum(slos) / len(slos) if slos else None,
            "slo_violation_percent_min": min(slos) if slos else None,
            "slo_violation_percent_max": max(slos) if slos else None,
            "paper_reference_target": PAPER_TARGETS.get(condition),
        })
    grouped_path = args.output_dir / "configuration_summary.csv"
    grouped_columns = ["condition", "quantized_layer_count", "run_count", "run_ids", "f1_percent_mean", "f1_percent_min", "f1_percent_max", "ttft_p95_s_mean", "ttft_p95_s_min", "ttft_p95_s_max", "slo_violation_percent_mean", "slo_violation_percent_min", "slo_violation_percent_max"]
    with grouped_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=grouped_columns)
        writer.writeheader()
        for row in grouped:
            row = dict(row)
            row["run_ids"] = ";".join(str(value) for value in row["run_ids"])
            writer.writerow({key: row.get(key) for key in grouped_columns})

    plot_path = args.output_dir / "quality_vs_p95_ttft.png"
    plot_results(summaries, plot_path)
    aggregate = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "paper_targets_are_reference_only": True,
        "runs": summaries,
        "configuration_summary": grouped,
        "derived_files": ["static_quantization_quality_latency_results.csv", "configuration_summary.csv", "quality_vs_p95_ttft.png"],
        "regeneration_command": "PYTHONPATH=swiftLLM python -m benchmark.analyze_static_quantization_quality_latency --manifest ... --runs ... --output-dir ...",
    }
    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

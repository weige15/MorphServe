"""Observation-only timing for the frozen v10 controller policy."""

from __future__ import annotations

import time
from typing import Any

from .analyze_crossover import counter_value_at, integrate, telemetry_segments, window_stats
from .closed_loop_controller import (
    FP16_POLICY_REFERENCE_BLOCKS,
    STABLE_AWQ,
    STABLE_FP16,
)
from .release_intent_controller import (
    PRIMARY_RELEASE_POLICY,
    RELEASE_CURRENT_UTILIZATION_MAX,
    RELEASE_HISTORY_WINDOW_S,
    RELEASE_PRESSURE_RATIO_MAX,
    RELEASE_PRIOR_WINDOW_S,
    RELEASE_RECENT_WINDOW_S,
    RELEASE_WAITING_MEAN_MAX,
    ReleaseIntentController,
)


class InstrumentedReleaseIntentController(ReleaseIntentController):
    """Reproduce v10 decisions while timing synchronous controller work.

    Timing fields are outputs only. No timing value participates in a policy
    condition, state transition, or action.
    """

    def __init__(self, measurement_start_ns: int):
        super().__init__(measurement_start_ns)
        self.timing_totals_ns = {
            "evaluate_wall": 0,
            "evaluate_cpu": 0,
            "window_wall": 0,
            "window_cpu": 0,
            "decision_bookkeeping_wall": 0,
            "decision_bookkeeping_cpu": 0,
            "trace_fields_wall": 0,
            "trace_fields_cpu": 0,
        }

    def evaluate(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return the frozen v10 decision plus observation-only timings."""
        evaluate_wall_start = time.perf_counter_ns()
        evaluate_cpu_start = time.process_time_ns()
        bookkeeping_wall_start = time.perf_counter_ns()
        bookkeeping_cpu_start = time.process_time_ns()

        now = float(sample["elapsed_s"])
        policy_row = {
            "elapsed_s": now,
            "waiting_q_depth": int(sample["waiting_q_depth"]),
            "num_decoding_gpu_blocks": int(sample["scheduler_used_kv_blocks"]),
            "num_gpu_blocks": FP16_POLICY_REFERENCE_BLOCKS,
            "logical_kv_utilization": (
                int(sample["scheduler_used_kv_blocks"]) / FP16_POLICY_REFERENCE_BLOCKS
            ),
            "preemption_count": int(sample["preemption_count"]),
        }
        self.policy_history.append(policy_row)
        bookkeeping_wall_ns = time.perf_counter_ns() - bookkeeping_wall_start
        bookkeeping_cpu_ns = time.process_time_ns() - bookkeeping_cpu_start

        window_wall_start = time.perf_counter_ns()
        window_cpu_start = time.process_time_ns()
        segments = telemetry_segments(self.policy_history, now)
        entry_stats = window_stats(segments, now, 3.0)
        release_stats = window_stats(segments, now, 10.0)
        recent_preemptions = int(policy_row["preemption_count"]) - counter_value_at(
            self.policy_history, "preemption_count", now - 10.0
        )
        base_window_wall_ns = time.perf_counter_ns() - window_wall_start
        base_window_cpu_ns = time.process_time_ns() - window_cpu_start

        entry_satisfied = bool(
            now >= 3.0
            and entry_stats["kv_ge095_fraction"] >= 0.80
            and entry_stats["waiting_ge4_fraction"] >= 0.50
            and int(policy_row["waiting_q_depth"]) >= 4
        )
        old_release_window_satisfied = bool(
            release_stats["waiting_mean"] <= 0.5
            and recent_preemptions == 0
            and release_stats["fp16_equiv_le070_fraction"] >= 0.90
        )
        old_release_satisfied = bool(
            self.state == STABLE_AWQ
            and self.last_entry_requested_s is not None
            and now >= self.last_entry_requested_s + 10.0
            and old_release_window_satisfied
        )
        old_action = None
        if self.state == STABLE_FP16 and entry_satisfied:
            old_action = "FP16_TO_AWQ_MARLIN_W4_16"
        elif old_release_satisfied:
            old_action = "AWQ_MARLIN_W4_16_TO_FP16"

        release_bookkeeping_wall_start = time.perf_counter_ns()
        release_bookkeeping_cpu_start = time.process_time_ns()
        if self.state == STABLE_AWQ:
            if self.first_awq_sample_s is None:
                self.first_awq_sample_s = now
            self.release_policy_history.append(self.policy_history[-1])
        bookkeeping_wall_ns += time.perf_counter_ns() - release_bookkeeping_wall_start
        bookkeeping_cpu_ns += time.process_time_ns() - release_bookkeeping_cpu_start

        window_wall_start = time.perf_counter_ns()
        window_cpu_start = time.process_time_ns()
        release_segments = telemetry_segments(self.release_policy_history, now)
        recent_start = now - RELEASE_RECENT_WINDOW_S
        prior_start = recent_start - RELEASE_PRIOR_WINDOW_S
        recent_waiting_mean = integrate(
            release_segments,
            lambda row: float(row["waiting_q_depth"]),
            start=recent_start,
            end=now,
        ) / RELEASE_RECENT_WINDOW_S
        recent_utilization_mean = integrate(
            release_segments,
            lambda row: float(row["logical_kv_utilization"]),
            start=recent_start,
            end=now,
        ) / RELEASE_RECENT_WINDOW_S
        prior_utilization_mean = integrate(
            release_segments,
            lambda row: float(row["logical_kv_utilization"]),
            start=prior_start,
            end=recent_start,
        ) / RELEASE_PRIOR_WINDOW_S
        utilization_ratio = (
            recent_utilization_mean / prior_utilization_mean
            if prior_utilization_mean > 0
            else None
        )
        release_recent_preemptions = int(sample["preemption_count"]) - counter_value_at(
            self.release_policy_history, "preemption_count", recent_start
        )
        release_window_wall_ns = time.perf_counter_ns() - window_wall_start
        release_window_cpu_ns = time.process_time_ns() - window_cpu_start
        window_wall_ns = base_window_wall_ns + release_window_wall_ns
        window_cpu_ns = base_window_cpu_ns + release_window_cpu_ns

        history_complete = bool(
            self.first_awq_sample_s is not None
            and now >= self.first_awq_sample_s + RELEASE_HISTORY_WINDOW_S
        )
        release_intent = bool(
            self.state == STABLE_AWQ
            and history_complete
            and int(sample["waiting_q_depth"]) == 0
            and recent_waiting_mean <= RELEASE_WAITING_MEAN_MAX
            and release_recent_preemptions == 0
            and float(sample["scheduler_used_kv_blocks"])
            / FP16_POLICY_REFERENCE_BLOCKS
            <= RELEASE_CURRENT_UTILIZATION_MAX
            and utilization_ratio is not None
            and utilization_ratio <= RELEASE_PRESSURE_RATIO_MAX
        )

        decision_bookkeeping_wall_start = time.perf_counter_ns()
        decision_bookkeeping_cpu_start = time.process_time_ns()
        decision = {
            "entry_window_elapsed_s": min(now, 3.0),
            "entry_waiting_mean": entry_stats["waiting_mean"],
            "entry_waiting_ge4_fraction": entry_stats["waiting_ge4_fraction"],
            "entry_kv_ge095_fraction": entry_stats["kv_ge095_fraction"],
            "entry_boolean": entry_satisfied,
            "release_window_elapsed_s": min(now, 10.0),
            "release_waiting_mean": release_stats["waiting_mean"],
            "release_fp16_equiv_le070_fraction": release_stats[
                "fp16_equiv_le070_fraction"
            ],
            "release_new_preemptions": recent_preemptions,
            "release_window_boolean": release_intent,
            "release_boolean": release_intent,
            "requested_action": (
                "AWQ_MARLIN_W4_16_TO_FP16"
                if release_intent
                else old_action
                if self.state != STABLE_AWQ
                else None
            ),
            "v9_old_release_window_boolean": old_release_window_satisfied,
            "v9_old_release_boolean": old_release_satisfied,
            "release_policy": PRIMARY_RELEASE_POLICY,
            "release_history_complete": history_complete,
            "release_recent_waiting_mean": recent_waiting_mean,
            "release_recent_preemptions": release_recent_preemptions,
            "release_current_fp16_reference_utilization": float(
                sample["scheduler_used_kv_blocks"]
            )
            / FP16_POLICY_REFERENCE_BLOCKS,
            "release_recent_fp16_reference_utilization_mean": recent_utilization_mean,
            "release_prior_fp16_reference_utilization_mean": prior_utilization_mean,
            "release_recent_to_prior_utilization_ratio": utilization_ratio,
            "release_intent_boolean": release_intent,
        }
        bookkeeping_wall_ns += time.perf_counter_ns() - decision_bookkeeping_wall_start
        bookkeeping_cpu_ns += time.process_time_ns() - decision_bookkeeping_cpu_start

        evaluate_wall_ns = time.perf_counter_ns() - evaluate_wall_start
        evaluate_cpu_ns = time.process_time_ns() - evaluate_cpu_start
        timings = {
            "controller_evaluate_wall_ns": evaluate_wall_ns,
            "controller_evaluate_cpu_ns": evaluate_cpu_ns,
            "controller_window_wall_ns": window_wall_ns,
            "controller_window_cpu_ns": window_cpu_ns,
            "controller_decision_bookkeeping_wall_ns": bookkeeping_wall_ns,
            "controller_decision_bookkeeping_cpu_ns": bookkeeping_cpu_ns,
            "controller_policy_history_length": len(self.policy_history),
            "controller_release_history_length": len(self.release_policy_history),
            "controller_policy_segment_count": len(segments),
            "controller_release_segment_count": len(release_segments),
        }
        self.timing_totals_ns["evaluate_wall"] += evaluate_wall_ns
        self.timing_totals_ns["evaluate_cpu"] += evaluate_cpu_ns
        self.timing_totals_ns["window_wall"] += window_wall_ns
        self.timing_totals_ns["window_cpu"] += window_cpu_ns
        self.timing_totals_ns["decision_bookkeeping_wall"] += bookkeeping_wall_ns
        self.timing_totals_ns["decision_bookkeeping_cpu"] += bookkeeping_cpu_ns
        decision.update(timings)
        return decision

    def trace_fields(self) -> dict[str, Any]:
        wall_start = time.perf_counter_ns()
        cpu_start = time.process_time_ns()
        fields = super().trace_fields()
        wall_ns = time.perf_counter_ns() - wall_start
        cpu_ns = time.process_time_ns() - cpu_start
        self.timing_totals_ns["trace_fields_wall"] += wall_ns
        self.timing_totals_ns["trace_fields_cpu"] += cpu_ns
        fields.update(
            {
                "controller_trace_fields_wall_ns": wall_ns,
                "controller_trace_fields_cpu_ns": cpu_ns,
                "controller_timing_totals_ns": dict(self.timing_totals_ns),
            }
        )
        return fields

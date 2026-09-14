"""Frozen release-intent controller for the release-side v10 experiment."""

from __future__ import annotations

from typing import Any

from .analyze_crossover import counter_value_at, integrate, telemetry_segments
from .closed_loop_controller import STABLE_AWQ, SustainedCompoundController


PRIMARY_RELEASE_POLICY = "quiet_pressure_drop_5v10"
RELEASE_HISTORY_WINDOW_S = 15.0
RELEASE_RECENT_WINDOW_S = 5.0
RELEASE_PRIOR_WINDOW_S = 10.0
RELEASE_WAITING_MEAN_MAX = 0.5
RELEASE_CURRENT_UTILIZATION_MAX = 1.25
RELEASE_PRESSURE_RATIO_MAX = 0.80


class ReleaseIntentController(SustainedCompoundController):
    """Preserve v7 entry exactly and replace only v9's failed AWQ exit rule."""

    def __init__(self, measurement_start_ns: int):
        super().__init__(measurement_start_ns)
        self.last_awq_stable_s: float | None = None
        self.first_awq_sample_s: float | None = None
        self.release_policy_history: list[dict[str, Any]] = []

    def evaluate(self, sample: dict[str, Any]) -> dict[str, Any]:
        decision = super().evaluate(sample)
        now = float(sample["elapsed_s"])
        if self.state == STABLE_AWQ:
            if self.first_awq_sample_s is None:
                self.first_awq_sample_s = now
            self.release_policy_history.append(self.policy_history[-1])
        segments = telemetry_segments(self.release_policy_history, now)
        recent_start = now - RELEASE_RECENT_WINDOW_S
        prior_start = recent_start - RELEASE_PRIOR_WINDOW_S
        recent_waiting_mean = integrate(
            segments,
            lambda row: float(row["waiting_q_depth"]),
            start=recent_start,
            end=now,
        ) / RELEASE_RECENT_WINDOW_S
        recent_utilization_mean = integrate(
            segments,
            lambda row: float(row["logical_kv_utilization"]),
            start=recent_start,
            end=now,
        ) / RELEASE_RECENT_WINDOW_S
        prior_utilization_mean = integrate(
            segments,
            lambda row: float(row["logical_kv_utilization"]),
            start=prior_start,
            end=recent_start,
        ) / RELEASE_PRIOR_WINDOW_S
        utilization_ratio = (
            recent_utilization_mean / prior_utilization_mean
            if prior_utilization_mean > 0
            else None
        )
        recent_preemptions = int(sample["preemption_count"]) - counter_value_at(
            self.release_policy_history, "preemption_count", recent_start
        )
        history_complete = bool(
            self.first_awq_sample_s is not None
            and now >= self.first_awq_sample_s + RELEASE_HISTORY_WINDOW_S
        )
        release_intent = bool(
            self.state == STABLE_AWQ
            and history_complete
            and int(sample["waiting_q_depth"]) == 0
            and recent_waiting_mean <= RELEASE_WAITING_MEAN_MAX
            and recent_preemptions == 0
            and float(sample["scheduler_used_kv_blocks"]) / 1768
            <= RELEASE_CURRENT_UTILIZATION_MAX
            and utilization_ratio is not None
            and utilization_ratio <= RELEASE_PRESSURE_RATIO_MAX
        )

        decision.update(
            {
                "v9_old_release_window_boolean": decision["release_window_boolean"],
                "v9_old_release_boolean": decision["release_boolean"],
                "release_policy": PRIMARY_RELEASE_POLICY,
                "release_history_complete": history_complete,
                "release_recent_waiting_mean": recent_waiting_mean,
                "release_recent_preemptions": recent_preemptions,
                "release_current_fp16_reference_utilization": float(
                    sample["scheduler_used_kv_blocks"]
                )
                / 1768,
                "release_recent_fp16_reference_utilization_mean": recent_utilization_mean,
                "release_prior_fp16_reference_utilization_mean": prior_utilization_mean,
                "release_recent_to_prior_utilization_ratio": utilization_ratio,
                "release_intent_boolean": release_intent,
                "release_window_boolean": release_intent,
                "release_boolean": release_intent,
                "requested_action": (
                    "AWQ_MARLIN_W4_16_TO_FP16"
                    if release_intent
                    else decision["requested_action"]
                    if self.state != STABLE_AWQ
                    else None
                ),
            }
        )
        return decision

    def complete_transition(
        self, transition_id: int, engine_trace: dict[str, Any], completed_ns: int
    ) -> None:
        direction = self.transitions[transition_id - 1]["direction"]
        super().complete_transition(transition_id, engine_trace, completed_ns)
        if direction == "FP16_TO_AWQ_MARLIN_W4_16":
            self.last_awq_stable_s = (
                completed_ns - self.measurement_start_ns
            ) / 1_000_000_000
            self.first_awq_sample_s = None
            self.release_policy_history = []

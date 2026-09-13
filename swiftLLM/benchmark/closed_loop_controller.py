"""Frozen v7 sustained-pressure controller for the v9 closed-loop experiment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .analyze_crossover import counter_value_at, telemetry_segments, window_stats


FP16_POLICY_REFERENCE_BLOCKS = 1768
PRIMARY_POLICY = "sustained_compound_pressure"
STABLE_FP16 = "FP16"
ENTRY_REQUESTED = "ENTRY_REQUESTED"
MORPHING_TO_AWQ = "MORPHING_TO_AWQ"
STABLE_AWQ = "AWQ"
RELEASE_REQUESTED = "RELEASE_REQUESTED"
RESTORING_TO_FP16 = "DRAINING_OR_RESTORING"
FAILED = "FAILED"


class SustainedCompoundController:
    """Causal state machine using the exact v7 time-weighted window helpers."""

    def __init__(self, measurement_start_ns: int):
        self.measurement_start_ns = measurement_start_ns
        self.state = STABLE_FP16
        self.policy_history: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.active_transition_id: int | None = None
        self.last_entry_requested_s: float | None = None
        self.error: str | None = None

    def evaluate(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Append one causal sample and return frozen rule statistics/action."""
        now = float(sample["elapsed_s"])
        policy_row = {
            "elapsed_s": now,
            "waiting_q_depth": int(sample["waiting_q_depth"]),
            # V7's telemetry signal is the scheduler's logical decoding-block
            # count. Physical allocator usage remains a separate safety field.
            "num_decoding_gpu_blocks": int(sample["scheduler_used_kv_blocks"]),
            "num_gpu_blocks": FP16_POLICY_REFERENCE_BLOCKS,
            # The v7 entry helper reads this field. Keep its policy denominator
            # fixed at 1768 rather than the runtime process's 1759 base.
            "logical_kv_utilization": (
                int(sample["scheduler_used_kv_blocks"]) / FP16_POLICY_REFERENCE_BLOCKS
            ),
            "preemption_count": int(sample["preemption_count"]),
        }
        self.policy_history.append(policy_row)
        segments = telemetry_segments(self.policy_history, now)
        entry_stats = window_stats(segments, now, 3.0)
        release_stats = window_stats(segments, now, 10.0)
        recent_preemptions = int(policy_row["preemption_count"]) - counter_value_at(
            self.policy_history, "preemption_count", now - 10.0
        )
        entry_satisfied = bool(
            now >= 3.0
            and entry_stats["kv_ge095_fraction"] >= 0.80
            and entry_stats["waiting_ge4_fraction"] >= 0.50
            and int(policy_row["waiting_q_depth"]) >= 4
        )
        release_window_satisfied = bool(
            release_stats["waiting_mean"] <= 0.5
            and recent_preemptions == 0
            and release_stats["fp16_equiv_le070_fraction"] >= 0.90
        )
        release_satisfied = bool(
            self.state == STABLE_AWQ
            and self.last_entry_requested_s is not None
            and now >= self.last_entry_requested_s + 10.0
            and release_window_satisfied
        )

        action = None
        if self.state == STABLE_FP16 and entry_satisfied:
            action = "FP16_TO_AWQ_MARLIN_W4_16"
        elif release_satisfied:
            action = "AWQ_MARLIN_W4_16_TO_FP16"

        return {
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
            "release_window_boolean": release_window_satisfied,
            "release_boolean": release_satisfied,
            "requested_action": action,
        }

    def request_transition(self, direction: str, requested_ns: int) -> int:
        if self.active_transition_id is not None:
            raise RuntimeError("a controller transition is already latched")
        if direction == "FP16_TO_AWQ_MARLIN_W4_16":
            if self.state != STABLE_FP16:
                raise RuntimeError(f"cannot enter AWQ from controller state {self.state}")
            self.state = ENTRY_REQUESTED
            self.last_entry_requested_s = (
                requested_ns - self.measurement_start_ns
            ) / 1_000_000_000
        elif direction == "AWQ_MARLIN_W4_16_TO_FP16":
            if self.state != STABLE_AWQ:
                raise RuntimeError(f"cannot restore FP16 from controller state {self.state}")
            self.state = RELEASE_REQUESTED
        else:
            raise ValueError(f"unsupported transition direction: {direction}")
        transition_id = len(self.transitions) + 1
        self.active_transition_id = transition_id
        self.transitions.append(
            {
                "controller_transition_id": transition_id,
                "direction": direction,
                "requested_ns": requested_ns,
                "requested_elapsed_s": (
                    requested_ns - self.measurement_start_ns
                ) / 1_000_000_000,
                "api_task_started_ns": None,
                "api_task_ended_ns": None,
                "model_transition_started_ns": None,
                "model_transition_ended_ns": None,
                "pending_or_drain_duration_s": None,
                "hot_transition_duration_s": None,
                "request_to_completion_duration_s": None,
                "result": "pending",
                "error": None,
                "engine_trace": None,
            }
        )
        return transition_id

    def mark_api_started(self, transition_id: int, timestamp_ns: int) -> None:
        event = self._active_event(transition_id)
        event["api_task_started_ns"] = timestamp_ns
        # Entry remains visibly requested until the v8 runtime state itself
        # reports MORPHING_TO_AWQ; the completed engine trace later supplies
        # the exact hot start. Restore has a real admission-drain pending state.
        if event["direction"] == "AWQ_MARLIN_W4_16_TO_FP16":
            self.state = RESTORING_TO_FP16

    def complete_transition(
        self, transition_id: int, engine_trace: dict[str, Any], completed_ns: int
    ) -> None:
        event = self._active_event(transition_id)
        event["api_task_ended_ns"] = completed_ns
        event["engine_trace"] = deepcopy(engine_trace)
        event["model_transition_started_ns"] = engine_trace.get("started_ns")
        event["model_transition_ended_ns"] = engine_trace.get("ended_ns")
        if event["model_transition_started_ns"] is not None:
            event["pending_or_drain_duration_s"] = (
                int(event["model_transition_started_ns"]) - int(event["requested_ns"])
            ) / 1_000_000_000
        if engine_trace.get("elapsed_ns") is not None:
            event["hot_transition_duration_s"] = int(engine_trace["elapsed_ns"]) / 1_000_000_000
        event["request_to_completion_duration_s"] = (
            completed_ns - int(event["requested_ns"])
        ) / 1_000_000_000
        event["result"] = str(engine_trace.get("status", "success"))
        self.state = (
            STABLE_AWQ
            if event["direction"] == "FP16_TO_AWQ_MARLIN_W4_16"
            else STABLE_FP16
        )
        self.active_transition_id = None

    def fail_transition(self, transition_id: int, error: BaseException, completed_ns: int) -> None:
        event = self._active_event(transition_id)
        event["api_task_ended_ns"] = completed_ns
        event["request_to_completion_duration_s"] = (
            completed_ns - int(event["requested_ns"])
        ) / 1_000_000_000
        event["result"] = "error"
        event["error"] = f"{type(error).__name__}: {error}"
        self.error = str(event["error"])
        self.state = FAILED
        self.active_transition_id = None

    def trace_fields(self) -> dict[str, Any]:
        event = self.transitions[-1] if self.transitions else None
        return {
            "controller_state": self.state,
            "active_transition_id": self.active_transition_id,
            "last_transition_id": event.get("controller_transition_id") if event else None,
            "transition_requested_timestamp_ns": event.get("requested_ns") if event else None,
            "transition_start_timestamp_ns": (
                event.get("model_transition_started_ns") if event else None
            ),
            "transition_end_timestamp_ns": (
                event.get("model_transition_ended_ns") if event else None
            ),
            "transition_pending_or_drain_duration_s": (
                event.get("pending_or_drain_duration_s") if event else None
            ),
            "transition_result": event.get("result") if event else None,
            "transition_error": event.get("error") if event else self.error,
        }

    def _active_event(self, transition_id: int) -> dict[str, Any]:
        if self.active_transition_id != transition_id:
            raise RuntimeError(f"transition {transition_id} is not the active latch")
        return self.transitions[transition_id - 1]

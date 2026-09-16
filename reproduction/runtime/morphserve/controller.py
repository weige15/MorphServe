"""Explicitly reconstructed MorphServe monitor/controller policy core."""

from __future__ import annotations

import math
from dataclasses import dataclass

SIGNALS = (
    "gpu_memory_usage",
    "kv_usage",
    "queue_depth",
    "queue_delay_s",
    "throughput_tokens_s",
    "ttft_s",
    "tpot_s",
)
DEFAULT_EMA_ALPHA = 0.25
PRESSURE_KV_USAGE = 0.85
PRESSURE_QUEUE_DELAY_S = 0.100
PRESSURE_SAMPLES = 3
RECOVERY_KV_USAGE = 0.65
RECOVERY_QUEUE_DELAY_S = 0.025
RECOVERY_SAMPLES = 5


@dataclass(frozen=True)
class ModeConfig:
    layers_per_action: int
    max_quantized_layers: int


MODE_CONFIGS = {
    "accuracy": ModeConfig(1, 4),
    "default": ModeConfig(2, 8),
    "performance": ModeConfig(4, 16),
}


class ServingMonitor:
    def __init__(self, alpha: float = DEFAULT_EMA_ALPHA):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.smoothed = None
        self.history = []

    def observe(self, timestamp_s: float, **metrics) -> dict:
        if set(metrics) != set(SIGNALS):
            raise ValueError(f"metrics must be exactly {SIGNALS}")
        values = {name: float(metrics[name]) for name in SIGNALS}
        if not math.isfinite(float(timestamp_s)) or any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("timestamp and metrics must be finite and non-negative")
        if values["gpu_memory_usage"] > 1 or values["kv_usage"] > 1:
            raise ValueError("usage metrics must be in [0, 1]")
        if self.smoothed is None:
            self.smoothed = values.copy()
        else:
            self.smoothed = {
                name: self.alpha * values[name] + (1 - self.alpha) * self.smoothed[name]
                for name in SIGNALS
            }
        sample = {"timestamp_s": float(timestamp_s), "raw": values, "smoothed": self.smoothed.copy()}
        self.history.append(sample)
        return sample


class MorphingController:
    def __init__(self, mode: str = "default"):
        if mode not in MODE_CONFIGS:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.config = MODE_CONFIGS[mode]
        self.quantized_layers = 0
        self.pressure_count = 0
        self.recovery_count = 0
        self.commands = []

    def update(self, sample: dict) -> dict:
        values = sample["smoothed"]
        if set(values) != set(SIGNALS) or any(not math.isfinite(float(values[name])) for name in SIGNALS):
            raise ValueError("invalid monitor sample")
        pressure = values["kv_usage"] >= PRESSURE_KV_USAGE or values["queue_delay_s"] >= PRESSURE_QUEUE_DELAY_S
        recovery = values["kv_usage"] <= RECOVERY_KV_USAGE and values["queue_delay_s"] <= RECOVERY_QUEUE_DELAY_S
        delta = 0
        reason = "neutral"
        if pressure:
            self.pressure_count += 1
            self.recovery_count = 0
            reason = "persistent_pressure"
            if self.pressure_count >= PRESSURE_SAMPLES and self.quantized_layers < self.config.max_quantized_layers:
                delta = min(
                    self.config.layers_per_action,
                    self.config.max_quantized_layers - self.quantized_layers,
                )
                self.pressure_count = 0
        elif recovery:
            self.recovery_count += 1
            self.pressure_count = 0
            reason = "persistent_recovery"
            if self.recovery_count >= RECOVERY_SAMPLES and self.quantized_layers > 0:
                delta = -min(self.config.layers_per_action, self.quantized_layers)
                self.recovery_count = 0
        else:
            self.pressure_count = 0
            self.recovery_count = 0
        self.quantized_layers += delta
        command = {
            "mode": self.mode,
            "reason": reason,
            "layer_delta": delta,
            "quantized_layers": self.quantized_layers,
            "expand_kv": delta > 0,
            "shrink_kv_before_restore": delta < 0,
            "pressure_count": self.pressure_count,
            "recovery_count": self.recovery_count,
            "sample_timestamp_s": sample["timestamp_s"],
        }
        self.commands.append(command)
        return command

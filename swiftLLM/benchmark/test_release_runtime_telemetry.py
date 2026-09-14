"""CPU checks for observation-only restore lifecycle telemetry."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import time
import unittest

from swiftllm.server.engine import Engine


class RestoreLifecycleTelemetryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def engine() -> Engine:
        engine = object.__new__(Engine)
        manager = SimpleNamespace(num_blocks=4, num_free_blocks=1)

        def restore_to_fp16(**_: object) -> dict:
            started_ns = time.perf_counter_ns()
            engine.model.runtime_precision_state = "FP16"
            engine.model.num_blocks = 2
            manager.num_blocks = 2
            manager.num_free_blocks = 0
            ended_ns = time.perf_counter_ns()
            return {
                "status": "success",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            }

        engine.initialized = True
        engine.engine_config = SimpleNamespace(
            enable_runtime_morphing=True,
            runtime_verify_kv=False,
            runtime_awq_target_blocks=4,
        )
        engine.model = SimpleNamespace(
            runtime_precision_state="AWQ_MARLIN_W4_16",
            base_num_blocks=2,
            num_blocks=4,
            gpu_block_manager=manager,
            restore_to_fp16=restore_to_fp16,
        )
        engine.scheduler = SimpleNamespace(
            waiting_q=[],
            running_q=[],
            swapped_q=[],
            num_decoding_gpu_blocks=3,
            num_gpu_blocks=4,
            admissions_paused=False,
        )
        engine._pending_transition = None
        engine._pending_restore_lifecycle = None
        engine._main_loop_running = True
        engine._transition_lock = asyncio.Lock()
        engine.runtime_transition_events = []
        engine.benchmark_swap_in_count = 0
        engine.benchmark_swap_out_count = 0

        async def run_now(func, *args, **kwargs):
            return func(*args, **kwargs)

        engine._run_on_model_async = run_now
        return engine

    async def test_drain_pause_feasibility_restore_and_resume_are_ordered(self) -> None:
        engine = self.engine()
        request = asyncio.create_task(engine.restore_to_fp16())
        await asyncio.sleep(0)
        self.assertFalse(await engine._service_pending_transition())
        lifecycle = engine._pending_restore_lifecycle
        self.assertTrue(engine.scheduler.admissions_paused)
        self.assertTrue(lifecycle["drain_required"])
        self.assertIsNotNone(lifecycle["admission_pause_ns"])
        self.assertIsNone(lifecycle["first_physical_shrink_legal_ns"])

        engine.model.gpu_block_manager.num_free_blocks = 2
        self.assertTrue(engine._mark_restore_feasible())
        self.assertTrue(await engine._service_pending_transition())
        trace = await request
        lifecycle = trace["restore_lifecycle"]
        ordered = [
            lifecycle["engine_restore_request_received_ns"],
            lifecycle["admission_pause_ns"],
            lifecycle["first_physical_shrink_legal_ns"],
            lifecycle["hot_restore_start_ns"],
            lifecycle["hot_restore_end_ns"],
            lifecycle["capacity_published_ns"],
            lifecycle["admission_resume_ns"],
        ]
        self.assertTrue(all(left <= right for left, right in zip(ordered, ordered[1:])))
        self.assertFalse(engine.scheduler.admissions_paused)
        self.assertEqual(engine.scheduler.num_gpu_blocks, 2)
        self.assertEqual(lifecycle["drain_duration_ns"], ordered[2] - ordered[1])


if __name__ == "__main__":
    unittest.main()

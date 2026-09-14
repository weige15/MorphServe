import asyncio
import functools
import time
from typing import AsyncGenerator

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.utils import GB

from .tokenization_engine import TokenizationEngine
from .structs import Request, RawRequest, StepOutput
from .scheduler import Scheduler

class Engine:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)
        self.initialized = False

        # The following fields will be created on `init_model()`
        self.model = None
        self.event_loop = None
        self.scheduler = None
        self.tokenization_engine = None

        self.untokenized_raw_requests: list[tuple[Request, str]] = []

        # Observational counters for benchmark telemetry. They do not affect
        # scheduler decisions or model execution.
        self.benchmark_swap_in_count = 0
        self.benchmark_swap_out_count = 0
        self.benchmark_batch_events: list[dict] | None = None
        self.benchmark_batch_index = 0

        # Manual transition requests are consumed only by the main loop at a
        # completed-forward boundary. There is deliberately no pressure rule.
        self._pending_transition = None
        self._pending_restore_lifecycle = None
        self._main_loop_running = False
        self._transition_lock = asyncio.Lock()
        self.runtime_transition_events: list[dict] = []

    async def _run_on_model_async(self, func, *args, **kwargs):
        """
        Run a function on the model asynchronously, and return the result
        """
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial)

    async def initialize(self):
        self.event_loop = asyncio.get_event_loop()

        print("[Engine] Initializing model...")
        self.model = LlamaModel(self.engine_config)

        print("[Engine] Loading weights...")
        self.model.load_weights()
        if self.engine_config.enable_runtime_morphing:
            print("[Engine] Preparing pinned FP16/AWQ runtime variants...")
            self.model.prepare_runtime_morphing()

        print("[Engine] Profiling kv blocks...")
        num_gpu_blocks = self.model.profile_num_blocks()
        num_cpu_blocks = self.engine_config.num_cpu_blocks
        block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        print(f"[Engine] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
        print(f"[Engine] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")

        print("[Engine] Allocating kv cache and swap...")
        self.model.init_kvcache_and_swap(num_gpu_blocks)

        print("[Engine] Initializing scheduler...")
        self.scheduler = Scheduler(self.model, self.engine_config, num_gpu_blocks)

        print("[Engine] Initializing tokenization engine...")
        self.tokenization_engine = TokenizationEngine.remote(self.engine_config)

        if (
            self.engine_config.enable_runtime_morphing
            and self.engine_config.runtime_awq_target_blocks < num_gpu_blocks
        ):
            raise ValueError("runtime AWQ target cannot be smaller than the FP16 base")

        print("[Engine] Model initialized")
        self.initialized = True

    def _transition_context(self) -> dict:
        scheduler = self.scheduler
        manager = self.model.gpu_block_manager
        active_requests = list(scheduler.running_q) + list(scheduler.swapped_q)
        return {
            "waiting_request_count": len(scheduler.waiting_q),
            "running_request_count": len(scheduler.running_q),
            "swapped_request_count": len(scheduler.swapped_q),
            "active_request_count": len(scheduler.running_q) + len(scheduler.swapped_q),
            "scheduler_used_kv_blocks": scheduler.num_decoding_gpu_blocks,
            "allocated_gpu_kv_blocks": manager.num_blocks - manager.num_free_blocks,
            "scheduler_visible_blocks": scheduler.num_gpu_blocks,
            "physical_blocks": self.model.num_blocks,
            "base_blocks": self.model.base_num_blocks,
            "extension_blocks": self.model.num_blocks - self.model.base_num_blocks,
            "active_request_states": [
                {
                    "benchmark_request_id": request.benchmark_request_id,
                    "request_id": request.request_id,
                    "prompt_len": request.prompt_len,
                    "output_len": request.get_cur_output_len(),
                    "next_input_position": request.prompt_len + request.get_cur_output_len() - 1,
                }
                for request in active_requests
            ],
        }

    async def _apply_transition(self, target: str) -> dict:
        before = self._transition_context()
        try:
            if target == "AWQ_MARLIN_W4_16":
                trace = await self._run_on_model_async(
                    self.model.morph_to_awq_w4_16,
                    self.engine_config.runtime_awq_target_blocks,
                    active_request_count=before["active_request_count"],
                    used_kv_blocks=before["allocated_gpu_kv_blocks"],
                    verify_kv=self.engine_config.runtime_verify_kv,
                )
            else:
                trace = await self._run_on_model_async(
                    self.model.restore_to_fp16,
                    active_request_count=before["active_request_count"],
                    used_kv_blocks=before["allocated_gpu_kv_blocks"],
                    verify_kv=self.engine_config.runtime_verify_kv,
                )
            # Publish admission capacity only after physical resize succeeds.
            lifecycle = getattr(self, "_pending_restore_lifecycle", None)
            if target == "FP16" and lifecycle is not None:
                lifecycle["hot_restore_start_ns"] = trace.get("started_ns")
                lifecycle["hot_restore_end_ns"] = trace.get("ended_ns")
                lifecycle["model_restore_returned_ns"] = time.perf_counter_ns()
            self.scheduler.num_gpu_blocks = self.model.num_blocks
            if target == "FP16" and lifecycle is not None:
                lifecycle["capacity_published_ns"] = time.perf_counter_ns()
            self.scheduler.admissions_paused = False
            if target == "FP16" and lifecycle is not None:
                lifecycle["admission_resume_ns"] = time.perf_counter_ns()
                lifecycle["context_at_admission_resume"] = self._transition_context()
                if lifecycle.get("drain_start_ns") is not None and lifecycle.get(
                    "first_physical_shrink_legal_ns"
                ) is not None:
                    lifecycle["drain_duration_ns"] = int(
                        lifecycle["first_physical_shrink_legal_ns"]
                    ) - int(lifecycle["drain_start_ns"])
                trace["restore_lifecycle"] = dict(lifecycle)
            trace["engine_context_before"] = before
            trace["engine_context_after"] = self._transition_context()
            self.runtime_transition_events.append(trace)
            return trace
        except Exception:
            # Never advertise more blocks than remain physically backed after
            # a failed transition or rollback.
            self.scheduler.num_gpu_blocks = self.model.num_blocks
            self.scheduler.admissions_paused = False
            raise

    async def _request_transition(self, target: str) -> dict:
        async with self._transition_lock:
            if not self.initialized:
                raise RuntimeError("engine must be initialized before runtime transition")
            if not self.engine_config.enable_runtime_morphing:
                raise RuntimeError("runtime morphing is disabled")
            if target not in ("FP16", "AWQ_MARLIN_W4_16"):
                raise ValueError(f"unsupported runtime precision target: {target}")
            if self.model.runtime_precision_state == target:
                return {
                    "schema_version": 1,
                    "status": "noop",
                    "precision_after": target,
                    "engine_context_after": self._transition_context(),
                }
            if not self._main_loop_running:
                allocated = (
                    self.model.gpu_block_manager.num_blocks
                    - self.model.gpu_block_manager.num_free_blocks
                )
                if target == "FP16" and allocated > self.model.base_num_blocks:
                    raise RuntimeError(
                        "direct restoration cannot drain active KV blocks; start the engine "
                        f"loop and drain to <= {self.model.base_num_blocks} first"
                    )
                return await self._apply_transition(target)
            if self._pending_transition is not None:
                pending_target, pending_future = self._pending_transition
                if pending_target == target:
                    return await asyncio.shield(pending_future)
                raise RuntimeError(
                    f"transition to {pending_target} is already pending; refusing conflicting request"
                )
            future = asyncio.get_running_loop().create_future()
            self._pending_transition = (target, future)
            if target == "FP16":
                self._pending_restore_lifecycle = {
                    "schema_version": 1,
                    "engine_restore_request_received_ns": time.perf_counter_ns(),
                    "context_at_restore_request": self._transition_context(),
                    "admission_pause_ns": None,
                    "drain_start_ns": None,
                    "first_physical_shrink_legal_ns": None,
                    "drain_end_ns": None,
                    "context_at_first_physical_shrink_legal": None,
                    "drain_required": None,
                    "drain_service_checks": 0,
                }
            return await asyncio.shield(future)

    async def morph_to_awq_w4_16(self) -> dict:
        """Manually request FP16 -> AWQ-Marlin W4-16 at a forward boundary."""
        return await self._request_transition("AWQ_MARLIN_W4_16")

    async def restore_to_fp16(self) -> dict:
        """Manually request AWQ-Marlin W4-16 -> FP16 at a forward boundary."""
        return await self._request_transition("FP16")

    def _mark_restore_feasible(self) -> bool:
        lifecycle = getattr(self, "_pending_restore_lifecycle", None)
        if lifecycle is None:
            return False
        allocated = (
            self.model.gpu_block_manager.num_blocks
            - self.model.gpu_block_manager.num_free_blocks
        )
        if allocated > self.model.base_num_blocks:
            return False
        if lifecycle["first_physical_shrink_legal_ns"] is None:
            timestamp_ns = time.perf_counter_ns()
            lifecycle["first_physical_shrink_legal_ns"] = timestamp_ns
            lifecycle["drain_end_ns"] = timestamp_ns
            lifecycle["context_at_first_physical_shrink_legal"] = self._transition_context()
        return True

    async def _service_pending_transition(self) -> bool:
        if self._pending_transition is None:
            return False
        target, future = self._pending_transition
        if target == "FP16":
            lifecycle = self._pending_restore_lifecycle
            if lifecycle["admission_pause_ns"] is None:
                timestamp_ns = time.perf_counter_ns()
                self.scheduler.admissions_paused = True
                lifecycle["admission_pause_ns"] = timestamp_ns
                lifecycle["drain_start_ns"] = timestamp_ns
                lifecycle["context_at_admission_pause"] = self._transition_context()
                lifecycle["drain_required"] = (
                    lifecycle["context_at_admission_pause"]["allocated_gpu_kv_blocks"]
                    > self.model.base_num_blocks
                )
            lifecycle["drain_service_checks"] += 1
            if not self._mark_restore_feasible():
                # Existing requests keep decoding; no new prefill or swap-in is
                # admitted until enough physical blocks can be retained in base.
                return False
        self._pending_transition = None
        try:
            result = await self._apply_transition(target)
        except Exception as exc:
            self.scheduler.num_gpu_blocks = self.model.num_blocks
            self.scheduler.admissions_paused = False
            if not future.done():
                future.set_exception(exc)
            if self.model.runtime_precision_state == "FAILED":
                raise
        else:
            if not future.done():
                future.set_result(result)
        finally:
            if target == "FP16":
                self._pending_restore_lifecycle = None
        return True

    async def add_request_and_stream(self, raw_request: RawRequest) -> AsyncGenerator[StepOutput, None]:
        """
        Add a raw request to the engine and stream the output of the request (streaming mode)
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        while True:
            step_output = await request.output_q.get()
            yield step_output
            request.output_q.task_done()
            if step_output.request.is_finished():
                break
    
    async def add_request_and_wait(self, raw_request: RawRequest) -> tuple[Request, list[int]]:
        """
        Add a raw request to the engine and wait for the completion (non-streaming mode)

        Return the output token ids
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        await request.finished_event.wait()
        return (request, request.output_token_ids)

    async def _tokenize_raw_request_event_loop(self):
        """
        Event loop for tokenizing raw requests
        """
        while True:
            if not self.untokenized_raw_requests:
                # No new raw requests, sleep for a bit
                await asyncio.sleep(0.002)
                continue

            # Tokenize the raw request in batch
            cur_untokenized_raw_requests = self.untokenized_raw_requests
            self.untokenized_raw_requests = []

            prompts = [prompt for _, prompt in cur_untokenized_raw_requests]
            prompt_token_ids = await self.tokenization_engine.batched_tokenize.remote(prompts)

            new_requests = []
            for (request, _), prompt_token_id in zip(cur_untokenized_raw_requests, prompt_token_ids):
                request.prompt_token_ids = prompt_token_id
                request.prompt_len = len(prompt_token_id)
                new_requests.append(request)

            eligible_time_ns = time.perf_counter_ns()
            for request in new_requests:
                request.benchmark_scheduler_eligible_time_ns = eligible_time_ns
            self.scheduler.on_requests_arrival(new_requests)
            await asyncio.sleep(0.001)  # yield the event loop
    
    async def _main_event_loop(self):
        """
        Event loop for forwarding the model
        """
        while True:
            await self._service_pending_transition()

            # Capture pressure immediately before scheduling. Observation is
            # opt-in and never participates in scheduler decisions.
            batch_observation = None
            if self.benchmark_batch_events is not None:
                snapshot = self.get_benchmark_snapshot()
                batch_observation = {
                    "schema_version": 1,
                    "batch_index": self.benchmark_batch_index,
                    "scheduling_timestamp_ns": time.perf_counter_ns(),
                    "waiting_queue_depth_before_scheduling": snapshot["waiting_q_depth"],
                    "running_request_count_before_scheduling": snapshot["running_q_count"],
                    "swapped_queue_depth_before_scheduling": snapshot["swapped_q_count"],
                    "safe_kv_blocks": snapshot["num_gpu_blocks"],
                    "used_kv_blocks_before_scheduling": snapshot["num_decoding_gpu_blocks"],
                    "allocated_gpu_blocks_before_scheduling": (
                        self.model.gpu_block_manager.num_blocks
                        - self.model.gpu_block_manager.num_free_blocks
                    ),
                    "free_gpu_blocks_before_scheduling": self.model.gpu_block_manager.num_free_blocks,
                    "logical_kv_utilization_before_scheduling": (
                        snapshot["num_decoding_gpu_blocks"] / snapshot["num_gpu_blocks"]
                        if snapshot["num_gpu_blocks"] else None
                    ),
                    "swap_in_count_before_scheduling": snapshot["swap_in_count"],
                    "swap_out_count_before_scheduling": snapshot["swap_out_count"],
                    "preemption_count_before_scheduling": snapshot["swap_out_count"],
                    "admissions_paused_before_scheduling": snapshot["admissions_paused"],
                    "pending_transition_target_before_scheduling": snapshot[
                        "pending_transition_target"
                    ],
                }

            # Get the next batch from the scheduler.
            cur_batch, cur_swap_in, cur_swap_out = self.scheduler.get_next_batch()
            if not cur_batch and not cur_swap_in and not cur_swap_out:
                # No new batch, sleep for a bit
                await asyncio.sleep(0.005)
                continue

            # Perform swap in/out. The counters are observational only; the
            # list materialization preserves the upstream call and ordering.
            swap_out_ids = []
            if cur_swap_out:
                swap_out_ids = [req.request_id for req in cur_swap_out]
                self.benchmark_swap_out_count += len(swap_out_ids)
                await self._run_on_model_async(
                    self.model.swap_out_seqs,
                    swap_out_ids
                )
            swap_in_ids = [req.request_id for req in cur_swap_in]
            if swap_in_ids:
                self.benchmark_swap_in_count += len(swap_in_ids)
                await self._run_on_model_async(
                    self.model.swap_in_seqs,
                    swap_in_ids
                )

            # Forward the model
            prefill_flags = [req.is_prefill_stage() for req in cur_batch]
            input_ids = [
                req.prompt_token_ids if is_prefill else [req.output_token_ids[-1]]
                for req, is_prefill in zip(cur_batch, prefill_flags)
            ]
            seq_ids = [req.request_id for req in cur_batch]
            decoding_seq_lens_list = [
                req.prompt_len + req.get_cur_output_len()
                for req in cur_batch
                if not req.is_prefill_stage()
            ]
            prefill_time_ns = time.perf_counter_ns()
            for req, is_prefill in zip(cur_batch, prefill_flags):
                if is_prefill and req.benchmark_first_prefill_time_ns is None:
                    req.benchmark_first_prefill_time_ns = prefill_time_ns
            output_tokens = await self._run_on_model_async(
                self.model.forward,
                input_ids,
                seq_ids,
                decoding_seq_lens_list
            )
            output_time_ns = time.perf_counter_ns()

            if batch_observation is not None:
                num_prefill_sequences = sum(prefill_flags)
                total_prefill_tokens = sum(
                    len(input_ids[index])
                    for index, is_prefill in enumerate(prefill_flags)
                    if is_prefill
                )
                forward_duration_ns = output_time_ns - prefill_time_ns
                batch_observation.update({
                    "timestamp_ns": prefill_time_ns,
                    "precision_state": self.model.runtime_precision_state,
                    "quantization_backend": (
                        self.model.weight.layers[0].quantization_backend
                        if self.model.weight.quantized_layer_count
                        else "fp16"
                    ),
                    "quantized_layer_count": self.model.weight.quantized_layer_count,
                    "batch_kind": (
                        "prefill" if num_prefill_sequences == len(cur_batch)
                        else "decode" if num_prefill_sequences == 0
                        else "mixed"
                    ),
                    "benchmark_request_ids": [req.benchmark_request_id for req in cur_batch],
                    "num_prefill_sequences": num_prefill_sequences,
                    "total_prefill_tokens": total_prefill_tokens,
                    "effective_gemm_m": total_prefill_tokens + len(decoding_seq_lens_list),
                    "num_decoding_sequences": len(decoding_seq_lens_list),
                    "forward_execution_duration_ns": forward_duration_ns,
                    "forward_execution_duration_s": forward_duration_ns / 1_000_000_000,
                    "prefill_execution_duration_ns": (
                        forward_duration_ns if num_prefill_sequences else None
                    ),
                    "prefill_execution_duration_s": (
                        forward_duration_ns / 1_000_000_000
                        if num_prefill_sequences else None
                    ),
                    "waiting_queue_depth_after_scheduling": len(self.scheduler.waiting_q),
                    "running_request_count_after_scheduling": len(self.scheduler.running_q),
                    "swapped_queue_depth_after_scheduling": len(self.scheduler.swapped_q),
                    "used_kv_blocks_after_scheduling": self.scheduler.num_decoding_gpu_blocks,
                    "allocated_gpu_blocks_after_forward": (
                        self.model.gpu_block_manager.num_blocks
                        - self.model.gpu_block_manager.num_free_blocks
                    ),
                    "free_gpu_blocks_after_forward": self.model.gpu_block_manager.num_free_blocks,
                    "logical_kv_utilization_after_scheduling": (
                        self.scheduler.num_decoding_gpu_blocks / self.scheduler.num_gpu_blocks
                        if self.scheduler.num_gpu_blocks else None
                    ),
                    "swap_in_count_this_step": len(swap_in_ids),
                    "swap_out_count_this_step": len(swap_out_ids),
                    "preemption_count_this_step": len(swap_out_ids),
                    "swap_in_count": self.benchmark_swap_in_count,
                    "swap_out_count": self.benchmark_swap_out_count,
                    "preemption_count": self.benchmark_swap_out_count,
                })
                self.benchmark_batch_events.append(batch_observation)
                self.benchmark_batch_index += 1

            # Deal with output tokens
            finished_req_ids = []
            for req, output_token in zip(cur_batch, output_tokens):
                req.output_token_ids.append(output_token)
                if req.benchmark_first_output_token_time_ns is None:
                    req.benchmark_first_output_token_time_ns = output_time_ns
                is_finished = req.is_finished()
                if is_finished:
                    # Capture the timestamp before publishing the final
                    # StepOutput; this remains observational only.
                    req.benchmark_completion_time_ns = output_time_ns
                req.output_q.put_nowait(StepOutput(
                    output_token,
                    req,
                    precision_state=self.model.runtime_precision_state,
                    input_position=req.prompt_len + req.get_cur_output_len() - 2,
                ))
                if is_finished:
                    finished_req_ids.append(req.request_id)
                    req.finished_event.set()
            await self._run_on_model_async(
                self.model.free_seqs_resources,
                finished_req_ids
            )
            # Inform the scheduler before recording the first-safe context so
            # physical allocation and active/scheduler request counts describe
            # the same completed-forward boundary.
            self.scheduler.on_batch_finish(cur_batch)
            if finished_req_ids:
                self._mark_restore_feasible()
    
    def start_benchmark_batch_observation(self) -> None:
        """Start a fresh, opt-in batch observation interval."""
        if not self.initialized:
            raise RuntimeError("engine must be initialized before batch observation")
        self.benchmark_batch_events = []
        self.benchmark_batch_index = 0
        self.benchmark_swap_in_count = 0
        self.benchmark_swap_out_count = 0

    def get_benchmark_batch_events(self) -> list[dict]:
        """Return a copy of recorded forward-batch observations."""
        return list(self.benchmark_batch_events or [])

    def get_benchmark_snapshot(self) -> dict[str, int]:
        """Return scheduler/runtime state for non-semantic benchmark telemetry."""
        scheduler = self.scheduler
        return {
            "waiting_q_depth": len(scheduler.waiting_q),
            "running_q_count": len(scheduler.running_q),
            "swapped_q_count": len(scheduler.swapped_q),
            "num_decoding_gpu_blocks": scheduler.num_decoding_gpu_blocks,
            "num_gpu_blocks": scheduler.num_gpu_blocks,
            "physical_gpu_blocks": self.model.num_blocks,
            "base_gpu_blocks": self.model.base_num_blocks,
            "extension_gpu_blocks": self.model.num_blocks - self.model.base_num_blocks,
            "precision_state": self.model.runtime_precision_state,
            "admissions_paused": scheduler.admissions_paused,
            "pending_transition_target": (
                self._pending_transition[0] if self._pending_transition is not None else None
            ),
            "swap_in_count": self.benchmark_swap_in_count,
            "swap_out_count": self.benchmark_swap_out_count,
        }

    async def start_all_event_loops(self):
        """
        Start all event loops
        """
        assert self.initialized, "Engine not initialized. Please call `initialize()` before starting the event loop."
        self._main_loop_running = True
        try:
            await asyncio.gather(
                self._tokenize_raw_request_event_loop(),
                self._main_event_loop()
            )
        finally:
            self._main_loop_running = False
            if self._pending_transition is not None:
                _, future = self._pending_transition
                if not future.done():
                    future.set_exception(RuntimeError("engine event loop stopped"))
                self._pending_transition = None
                self._pending_restore_lifecycle = None

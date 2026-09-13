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

        print("[Engine] Model initialized")
        self.initialized = True
    
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
                    "precision_state": (
                        "FP16"
                        if self.engine_config.quantized_layer_count == 0
                        else "AWQ-Marlin W4-16"
                        if self.engine_config.quantization_backend == "awq_marlin"
                        and self.engine_config.quantized_layer_count == 16
                        else f"{self.engine_config.quantization_backend}:{self.engine_config.quantized_layer_count}"
                    ),
                    "quantization_backend": self.engine_config.quantization_backend,
                    "quantized_layer_count": self.engine_config.quantized_layer_count,
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
                req.output_q.put_nowait(StepOutput(output_token, req))
                if is_finished:
                    finished_req_ids.append(req.request_id)
                    req.finished_event.set()
            await self._run_on_model_async(
                self.model.free_seqs_resources,
                finished_req_ids
            )
            
            # Inform the scheduler
            self.scheduler.on_batch_finish(cur_batch)
    
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
            "swap_in_count": self.benchmark_swap_in_count,
            "swap_out_count": self.benchmark_swap_out_count,
        }

    async def start_all_event_loops(self):
        """
        Start all event loops
        """
        assert self.initialized, "Engine not initialized. Please call `initialize()` before starting the event loop."
        await asyncio.gather(
            self._tokenize_raw_request_event_loop(),
            self._main_event_loop()
        )

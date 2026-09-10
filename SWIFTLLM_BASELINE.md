# SwiftLLM Full-Precision Baseline

Status: **PASS — online streaming smoke test completed**

This is the untouched upstream control/data-plane baseline used for Phase 1. The
repository's actual SwiftLLM tree is `swiftLLM/` (the top-level `README.md` is
only a placeholder). The inspected files are the vendored upstream files under
`swiftLLM/`, including `examples/online.py`, `swiftllm/server/{engine,scheduler,structs}.py`,
and `swiftllm/engine_config.py`.

## Source and model

- SwiftLLM upstream repository: https://github.com/interestingLSY/swiftLLM.git
- Upstream source commit: `682cf9a28f97f7490409981a2f181528f377eb5d`
- Vendoring commit: `23f0da8`
- Model path used (local HF snapshot):
  `/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Model revision: `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Model: Meta Llama 3.1 8B, four unquantized `.safetensors` shards; `config.json`
  reports 32 layers, 32 query heads, 8 KV heads, hidden size 4096.
- “Full precision” here means the existing unquantized FP16 execution path:
  `LlamaModel.load_weights()` explicitly loads `torch.float16` (the model
  metadata itself reports `bfloat16`). No quantization, layer swapping, KV
  resizing, or scheduler changes were made.

## Verified environment

Command interpreter: `/nfs/home/s314511048/.venv/bin/python` (Python 3.12.3).
The host has seven NVIDIA GeForce RTX 3090 GPUs; GPU 3 was selected because it
was idle for the run.

- NVIDIA driver `580.159.03`; CUDA toolkit / `nvcc` `12.4.99`
- `CUDA_VISIBLE_DEVICES=3` (SwiftLLM sees one RTX 3090, compute capability 8.6)
- PyTorch `2.5.1+cu121`, Torch CUDA `12.1`
- Triton `3.1.0`
- Transformers `5.16.1`, Ray `2.55.1`, safetensors `0.8.0`
- FastAPI `0.136.3`, Uvicorn `0.48.0`
- `vllm-flash-attn==2.6.2` (installed with `--no-deps`; its runtime import and
  attention call were tested with the existing Torch 2.5.1 environment)

Initially the required top-level `vllm_flash_attn` module was absent. The only
necessary dependency repair was installing that wheel. The C++ extension's
upstream `pip install -e csrc` command fails under this environment because
setuptools' isolated build cannot import Torch; the upstream extension itself
builds successfully with the direct `setup.py build_ext --inplace` command below.

## Engine configuration and profile

Exactly the values from `swiftLLM/examples/online.py` were used:

```text
EngineConfig(
  model_path=<local Llama-3.1-8B snapshot>, use_dummy=False,
  block_size=16, gpu_mem_utilization=0.99, num_cpu_blocks=1024,
  max_seqs_in_block_table=128, max_blocks_per_seq=3072,
  max_batch_size=4, max_tokens_in_batch=1024,
)
```

Initialization output from the real model run:

```text
[Model.profile] GPU total memory: 23.56 GB, runtime peak memory: 15.74 GB
[Engine] Number of GPU blocks: 3880 (7.58 GB)
[Engine] Number of CPU blocks: 1024 (2.00 GB)
[Engine] Model initialized
```

The profiled value is therefore `num_gpu_blocks=3880`; it was not manually
substituted. With this Llama configuration, one KV token is 131072 bytes and a
16-token block is 2 MiB, giving a GPU KV-cache capacity of `3880 * 16 = 62080`
token slots (plus `1024 * 16 = 16384` CPU swap token slots).

## Exact reproduction and smoke evidence

```bash
VENV=/nfs/home/s314511048/.venv
MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
$VENV/bin/python -m pip install --no-deps 'vllm-flash-attn==2.6.2'
cd swiftLLM/csrc
CUDA_VISIBLE_DEVICES=3 $VENV/bin/python setup.py build_ext --inplace
cd ..
PYTHONPATH="$PWD:$PWD/csrc" CUDA_VISIBLE_DEVICES=3 \
  $VENV/bin/python -u examples/online.py --model-path "$MODEL" --streaming
```

The command used the existing `Engine.initialize()` and
`Engine.start_all_event_loops()` path. It generated **80 streamed tokens**
across the four requests in `online.py` (5, 10, 15, and 50 tokens). The output
contained per-token latency arrays, not just a final non-streaming response;
representative completed output was:

```text
Output:  если camouflage relic relic.FileInputStream
Token latencies (ms): [11731.6, 4657.4, 40.0, 39.0, 38.2]
```

The other three requests completed with 10, 15, and 50 latency entries. The
same exact command was rerun immediately afterward and again produced all four
completed streams and the same `3880`-block profile. This proves model loading,
profiling, KV allocation, tokenization through Ray, control-plane scheduling,
GPU forward passes, and streaming delivery all ran successfully.

## Untouched-baseline audit

- `git diff -- swiftLLM/swiftllm/server swiftLLM/examples/online.py` was empty
  before this report was written.
- The existing FCFS admission, preemption, swapping, and batch semantics in
  `scheduler.py` were not modified.
- No MorphServe scheduler, quantization, layer swapping, KV resizing, or
  performance optimization was added.
- The broad shared environment's `pip check` reports unrelated pre-existing
  `vllm`/`xformers` dependency mismatches; none are imported by this verified
  SwiftLLM path. The exact versions and commands above reproduce the successful
  run.

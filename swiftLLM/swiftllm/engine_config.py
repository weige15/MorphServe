import dataclasses
import argparse

@dataclasses.dataclass
class EngineConfig:
    """
    Configuration for the SwiftLLM engine.
    """
    
    # Model loading parameters
    model_path: str
    use_dummy: bool

    # PagedAttention-related parameters
    block_size: int
    gpu_mem_utilization: float
    num_cpu_blocks: int
    max_seqs_in_block_table: int
    max_blocks_per_seq: int

    # Scheduling-related parameters
    max_batch_size: int
    max_tokens_in_batch: int

    # Optional static layer-quantization controls.  A value of zero preserves
    # the upstream FP16 path; positive values quantize decoder layers in the
    # fixed front-to-back order used by the static-quantization proxy benchmark.
    quantized_layer_count: int = 0
    quantization_backend: str = "nf4_bitsandbytes"
    quantized_model_path: str | None = None

    # Explicit manual two-state runtime substrate. This never enables a
    # workload-pressure controller; transitions are requested through Engine.
    enable_runtime_morphing: bool = False
    runtime_awq_target_blocks: int = 4170
    runtime_verify_kv: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """
        Add CLI arguments for the engine configuration
        """
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the model directory (currently SwiftLLM does not support downloading from HuggingFace, so please download in advance)",
        )
        parser.add_argument(
            "--use-dummy",
            action="store_true",
            help="Use dummy weights (mainly for profiling)",
        )

        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="Block size for PagedAttention",
        )
        parser.add_argument(
            "--gpu-mem-utilization",
            type=float,
            default=0.97,
            help="Fraction of GPU memory to be used",
        )
        parser.add_argument(
            "--num-cpu-blocks",
            type=int,
            default=2048,
            help="Number of CPU blocks",
        )
        parser.add_argument(
            "--max-seqs-in-block-table",
            type=int,
            default=4096,
            help="Maximum number of sequences in the block table",
        )
        parser.add_argument(
            "--max-blocks-per-seq",
            type=int,
            default=32768,
            help="Maximum number of blocks per sequence",
        )

        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=512,
            help="Maximum batch size",
        )
        parser.add_argument(
            "--max-tokens-in-batch",
            type=int,
            default=32768,
            help="Maximum number of tokens in a batch",
        )
        parser.add_argument(
            "--quantized-layer-count",
            type=int,
            default=0,
            choices=range(33),
            help="Number of front-to-back decoder layers to run with the W4 backend",
        )
        parser.add_argument(
            "--quantization-backend",
            choices=("nf4_bitsandbytes", "awq_marlin"),
            default="nf4_bitsandbytes",
        )
        parser.add_argument(
            "--quantized-model-path",
            type=str,
            help="Offline AutoAWQ checkpoint required by the AWQ-Marlin backend",
        )
        parser.add_argument(
            "--enable-runtime-morphing",
            action="store_true",
            help="Prepare explicit manual FP16 <-> AWQ-Marlin W4-16 transitions",
        )
        parser.add_argument(
            "--runtime-awq-target-blocks",
            type=int,
            default=4170,
            help="Measured safe dynamic KV target after a manual AWQ transition",
        )
        parser.add_argument(
            "--runtime-verify-kv",
            action="store_true",
            help="Hash sampled active logical KV blocks across manual resize operations",
        )
        
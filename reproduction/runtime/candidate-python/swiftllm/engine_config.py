import argparse
import dataclasses
import json


@dataclasses.dataclass
class EngineConfig:
    org_model_path: str
    use_dummy: bool = False
    quantized_model_path: str | None = None
    quantized_q_config: dict = dataclasses.field(
        default_factory=lambda: {"w_bit": 4, "q_group_size": 128, "zero_point": True}
    )
    block_size: int = 16
    gpu_mem_utilization: float = 0.97
    gpu_kv_cache_threshold: float = 0.85
    num_layers_to_quantize: int = 1
    num_layers_to_quantize_max: int = 0
    quant_serve: bool = False
    awq_mode: bool = False
    awq_half_mode: bool = False
    num_cpu_blocks: int = 2048
    max_seqs_in_block_table: int = 4096
    max_blocks_per_seq: int = 32768
    max_batch_size: int = 512
    max_tokens_in_batch: int = 32768
    req_preempt: int = 0
    block_swap: int = 0

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model-path", dest="org_model_path", required=True)
        parser.add_argument("--use-dummy", action="store_true")
        parser.add_argument("--quantized-model-path")
        parser.add_argument(
            "--quantized-q-config",
            type=json.loads,
            default={"w_bit": 4, "q_group_size": 128, "zero_point": True},
        )
        parser.add_argument("--block-size", type=int, default=16)
        parser.add_argument("--gpu-mem-utilization", type=float, default=0.97)
        parser.add_argument("--gpu-kv-cache-threshold", type=float, default=0.85)
        parser.add_argument("--num-layers-to-quantize", type=int, default=1)
        parser.add_argument("--num-layers-to-quantize-max", type=int, default=0)
        parser.add_argument("--quant-serve", action="store_true")
        parser.add_argument("--awq-mode", action="store_true")
        parser.add_argument("--awq-half-mode", action="store_true")
        parser.add_argument("--num-cpu-blocks", type=int, default=2048)
        parser.add_argument("--max-seqs-in-block-table", type=int, default=4096)
        parser.add_argument("--max-blocks-per-seq", type=int, default=32768)
        parser.add_argument("--max-batch-size", type=int, default=512)
        parser.add_argument("--max-tokens-in-batch", type=int, default=32768)

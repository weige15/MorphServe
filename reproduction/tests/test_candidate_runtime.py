import argparse
import unittest

from swiftllm import EngineConfig


class CandidateRuntimeNormalizationTests(unittest.TestCase):
    def test_cli_builds_complete_no_morph_config(self):
        parser = argparse.ArgumentParser()
        EngineConfig.add_cli_args(parser)
        config = EngineConfig(**vars(parser.parse_args(["--model-path", "/tmp/model"])))

        self.assertEqual(config.org_model_path, "/tmp/model")
        self.assertFalse(config.quant_serve)
        self.assertFalse(config.awq_mode)
        self.assertIsNone(config.quantized_model_path)
        self.assertEqual(config.quantized_q_config["w_bit"], 4)
        self.assertEqual(config.gpu_kv_cache_threshold, 0.85)


if __name__ == "__main__":
    unittest.main()

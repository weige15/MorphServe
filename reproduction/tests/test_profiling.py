import json
import tempfile
import unittest
from pathlib import Path

from morphserve.profiling import rank_layers, save_profile


class RankLayersTests(unittest.TestCase):
    def test_recomputes_mds_for_every_remaining_candidate(self):
        lts = {0: 0.8, 1: 0.8, 2: 0.8}
        lrs = {0: 0.8, 1: 0.8, 2: 0.8}
        conditioned_mds = {
            ((), 0): 0.9,
            ((), 1): 0.8,
            ((), 2): 0.7,
            ((0,), 1): 0.1,
            ((0,), 2): 0.9,
            ((0, 2), 1): 0.8,
        }
        calls = []

        def mds(layer, quantized):
            key = (tuple(quantized), layer)
            calls.append(key)
            return conditioned_mds[key]

        profile = rank_layers(lts, lrs, mds)

        self.assertEqual(profile["order"], [0, 2, 1])
        self.assertEqual(calls, list(conditioned_mds))
        self.assertEqual(
            [round(candidate["lis"], 8) for candidate in profile["steps"][0]["candidates"]],
            [0.85, 0.8, 0.75],
        )
        self.assertEqual(profile["weights"], {"lts": 0.25, "lrs": 0.25, "mds": 0.5})
        static_order = sorted(
            lts,
            key=lambda layer: (-(0.25 * lts[layer] + 0.25 * lrs[layer] + 0.5 * conditioned_mds[((), layer)]), layer),
        )
        self.assertEqual(static_order, [0, 1, 2])
        self.assertNotEqual(profile["order"], static_order)

    def test_ties_use_lowest_layer_index(self):
        profile = rank_layers({7: 0.5, 3: 0.5}, {7: 0.5, 3: 0.5}, lambda _layer, _quantized: 0.5)

        self.assertEqual(profile["order"], [3, 7])
        self.assertEqual(profile["tie_break"], "lowest_layer_index")

    def test_rejects_invalid_local_metrics(self):
        with self.assertRaisesRegex(ValueError, "same layer indices"):
            rank_layers({0: 0.5}, {1: 0.5}, lambda _layer, _quantized: 0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_layers({0: float("nan")}, {0: 0.5}, lambda _layer, _quantized: 0.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_layers({0: 0.5}, {0: 0.5}, lambda _layer, _quantized: float("inf"))

    def test_saves_complete_profile_as_canonical_json(self):
        profile = rank_layers({0: 0.5}, {0: 0.5}, lambda _layer, _quantized: 0.75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            save_profile(profile, path)
            text = path.read_text()

        self.assertEqual(json.loads(text), profile)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(text.splitlines()[1].strip(), '"algorithm": "MorphServe Algorithm 1 conditioned-MDS greedy LIS",')


if __name__ == "__main__":
    unittest.main()

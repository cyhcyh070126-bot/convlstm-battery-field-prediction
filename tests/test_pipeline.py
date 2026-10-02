"""Small functional checks; generated fixtures are not research data."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from python.checkpoints import load_checkpoint
from python.data.dataset import BatteryDataset
from python.evaluation.predict import require_fresh_output, rollout
from python.models import PaperModel
from python.train.engine import make_split, sequence_loss
from python.train.losses import HybridLoss


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loss_and_autoregressive_backward(self):
        torch.manual_seed(7)
        model = PaperModel()
        inputs = torch.rand(1, 5, 9, 16, 16)
        targets = torch.rand(1, 2, 3, 16, 16)
        loss = sequence_loss(model, inputs, targets, HybridLoss(), epsilon=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.conv3d.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.conv_lstm.cell_list[0].conv.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            prediction = rollout(model, inputs, 2)
        self.assertEqual(tuple(prediction.shape), (1, 2, 3, 16, 16))
        self.assertTrue(torch.all((prediction >= 0) & (prediction <= 1)))
        self.assertAlmostEqual(HybridLoss()(targets[:, 0], targets[:, 0]).item(), 0.0, places=6)

    def test_dataset_field_order_and_spatial_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            for subfolder in ["1_Concentration", "2_Stress", "3_Voronoi_Geometry", "C-rate"]:
                (case / subfolder).mkdir(parents=True)
            for name, value in [("frame_1.png", 10), ("frame_2.png", 20), ("frame_10.png", 100)]:
                Image.fromarray(np.full((16, 16, 3), value, np.uint8)).save(case / "1_Concentration" / name)
                Image.fromarray(np.full((16, 16, 3), value + 30, np.uint8)).save(case / "2_Stress" / name)
            Image.fromarray(np.full((16, 16, 3), 150, np.uint8)).save(case / "3_Voronoi_Geometry" / "05_orientation.png")
            Image.fromarray(np.full((16, 16, 3), 200, np.uint8)).save(case / "C-rate" / "rate.png")
            dataset = BatteryDataset([case], "stress", 2, 1, 16, 16, augment=True)
            inputs, target = dataset[0]
            self.assertEqual(tuple(inputs.shape), (2, 9, 16, 16))
            self.assertAlmostEqual(inputs[0, 0, 0, 0].item(), 40 / 255, places=6)
            self.assertAlmostEqual(inputs[1, 0, 0, 0].item(), 50 / 255, places=6)
            self.assertAlmostEqual(target[0, 0, 0, 0].item(), 130 / 255, places=6)
            self.assertAlmostEqual(inputs[0, 3, 0, 0].item(), 150 / 255, places=6)
            self.assertAlmostEqual(inputs[0, 6, 0, 0].item(), 200 / 255, places=6)
            (case / "2_Stress" / "frame_2.png").write_bytes(b"invalid png")
            with self.assertRaisesRegex(ValueError, "Unable to decode"):
                dataset[0]

    def test_case_split_prevents_overlap_and_survives_json(self):
        cases = [Path(str(i)) for i in range(10)]
        train, validation, split = make_split(cases, 0.8, 42)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual((len(train), len(validation)), (8, 2))
        reloaded = json.loads(json.dumps(split))
        self.assertEqual(make_split(cases, 0.9, 999, reloaded)[2], split)
        with self.assertRaises(ValueError):
            make_split(cases, existing={"train": ["0"], "validation": ["0"]})

    def test_checkpoint_legacy_and_field_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            model = PaperModel()
            torch.save(model.state_dict(), path)
            self.assertEqual(load_checkpoint(path, PaperModel()), {})
            torch.save({"model_state_dict": model.state_dict(), "config": {"field": "stress"}}, path)
            with self.assertRaisesRegex(ValueError, "trained for stress"):
                load_checkpoint(path, PaperModel(), field="concentration")

    def test_evaluation_does_not_mix_old_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation"
            require_fresh_output(path)
            path.mkdir()
            require_fresh_output(path)
            (path / "rollout.gif").write_bytes(b"previous output")
            with self.assertRaisesRegex(FileExistsError, "new or empty"):
                require_fresh_output(path)


if __name__ == "__main__":
    unittest.main()

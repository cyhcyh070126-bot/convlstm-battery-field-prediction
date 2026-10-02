"""Small functional checks; generated fixtures are not research data."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from python.checkpoints import load_checkpoint
from python.data.dataset import BatteryDataset, STATIC_CACHE_SIZE
from python.evaluation.predict import require_fresh_output, rollout
from python.models import PaperModel
from python.train.engine import build_parser, make_split, require_fresh_training_output, run, sequence_loss
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
            with self.assertWarnsRegex(UserWarning, "cannot confirm"):
                load_checkpoint(path, PaperModel(), field="stress")
            torch.save({"model_state_dict": model.state_dict(), "config": {"field": "stress"}}, path)
            with self.assertRaisesRegex(ValueError, "trained for stress"):
                load_checkpoint(path, PaperModel(), field="concentration")

    def test_released_checkpoint_cannot_be_mislabeled_as_stress(self):
        source = Path(__file__).resolve().parents[1] / "checkpoints" / "mse-ssim-pretrained.pth"
        with tempfile.TemporaryDirectory() as directory:
            # The guard must survive a filename change and must not fabricate
            # training metadata for these legacy tensor-only weights.
            renamed = Path(directory) / "my_weights.pt"
            shutil.copyfile(source, renamed)
            self.assertEqual(load_checkpoint(renamed, PaperModel(), field="concentration"), {})
            with self.assertRaisesRegex(ValueError, "trained for concentration"):
                load_checkpoint(renamed, PaperModel(), field="stress")

    def test_patch_values_match_full_images_and_static_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = []
            grid = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3)
            for index in range(STATIC_CACHE_SIZE + 1):
                case = Path(directory) / f"case_{index}"
                for subfolder in ["1_Concentration", "3_Voronoi_Geometry", "C-rate"]:
                    (case / subfolder).mkdir(parents=True)
                for frame in range(2):
                    Image.fromarray(((grid + frame + index) % 256).astype(np.uint8)).save(
                        case / "1_Concentration" / f"frame_{frame}.png")
                Image.fromarray(((grid + 17) % 256).astype(np.uint8)).save(
                    case / "3_Voronoi_Geometry" / "05_orientation.png")
                Image.fromarray(((grid + 31) % 256).astype(np.uint8)).save(
                    case / "C-rate" / "rate.png")
                cases.append(case)
            full = BatteryDataset(cases, input_length=1, predict_length=1,
                                  image_size=16, patch_size=16, use_patches=False)
            patches = BatteryDataset(cases, input_length=1, predict_length=1,
                                     image_size=16, patch_size=8)
            for case_index in range(len(cases)):
                inputs, targets = full[case_index]
                for patch in range(4):
                    row, col = divmod(patch, 2)
                    cropped_inputs, cropped_targets = patches[case_index * 4 + patch]
                    region = (..., slice(row * 8, row * 8 + 8), slice(col * 8, col * 8 + 8))
                    self.assertTrue(torch.equal(cropped_inputs, inputs[region]))
                    self.assertTrue(torch.equal(cropped_targets, targets[region]))
            self.assertEqual(len(patches.static_cache), STATIC_CACHE_SIZE)
            self.assertNotIn(0, patches.static_cache)
            # Reloading an evicted case must reproduce the same channels.
            restored, _ = patches[0]
            self.assertTrue(torch.equal(restored, full[0][0][..., :8, :8]))

    def test_training_does_not_overwrite_partial_or_unrelated_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training"
            require_fresh_training_output(path)
            path.mkdir()
            require_fresh_training_output(path)
            artifact = path / "best_model.pt"
            artifact.write_bytes(b"valuable weights without a history.csv")
            with self.assertRaisesRegex(FileExistsError, "new or empty"):
                require_fresh_training_output(path)
            self.assertEqual(artifact.read_bytes(), b"valuable weights without a history.csv")

    def test_invalid_training_numbers_fail_before_loading_data(self):
        for option, value in [("--max-batches", "0"), ("--num-workers", "-1"),
                              ("--learning-rate", "nan"), ("--ssim-weight", "inf"),
                              ("--epsilon-decay", "nan")]:
            with self.subTest(option=option):
                args = build_parser(sequence=True).parse_args([
                    "--data-dir", "missing_data", "--output-dir", "unused_output",
                    "--checkpoint", "missing.pt", option, value,
                ])
                with self.assertRaises(ValueError):
                    run(args)
        for value in [float("nan"), float("inf"), -0.1]:
            with self.assertRaises(ValueError):
                HybridLoss(value)

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

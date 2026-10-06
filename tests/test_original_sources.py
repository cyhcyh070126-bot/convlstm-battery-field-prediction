"""Numerical differential checks against the user's preserved source code."""
import ast
import importlib.util
import json
import hashlib
import sys
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from pytorch_msssim import ssim

from python.models import PaperModel
from python.train.losses import HybridLoss
from python.data.dataset import BatteryDataset, load_rgb

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class OriginalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_complete_manifest_and_original_hashes(self):
        manifest = json.loads((ROOT/'docs/complete-source-manifest.json').read_text(encoding='utf8'))
        self.assertEqual(len(manifest['files']), 40)
        self.assertEqual(len(manifest['excluded']), 8)
        for entry in manifest['files']:
            self.assertEqual(hashlib.sha256((ROOT/entry['archive']).read_bytes()).hexdigest(), entry['sha256'], entry['path'])
            self.assertEqual(hashlib.sha256((ROOT/entry['research_copy']).read_bytes()).hexdigest(), entry['research_sha256'], entry['path'])
            if entry['path'].endswith('.py'):
                ast.parse((ROOT/entry['research_copy']).read_bytes())

    def test_original_model_output_loss_and_gradients(self):
        namespace = definitions(ROOT/'source_archive/convlstm.py', {'ConvLSTMCell','ConvLSTM'}, {'torch':torch,'nn':nn})
        namespace['ssim'] = ssim
        definitions(ROOT/'source_archive/训练代码/单帧预测模型/train_pytorch.py', {'PaperModel','HybridLoss'},namespace)
        torch.manual_seed(17)
        original = namespace['PaperModel']()
        packaged = PaperModel()
        packaged.load_state_dict(original.state_dict(),strict=True)
        inputs = torch.rand(1,5,9,16,16)
        targets = torch.rand(1,3,16,16)
        a,b = original(inputs), packaged(inputs)
        self.assertTrue(torch.equal(a,b))
        la,lb = namespace['HybridLoss']()(a,targets), HybridLoss()(b,targets)
        self.assertTrue(torch.equal(la,lb))
        la.backward(); lb.backward()
        for (key,x),(key_b,y) in zip(original.named_parameters(),packaged.named_parameters()):
            self.assertEqual(key,key_b)
            self.assertTrue(torch.equal(x.grad,y.grad),key)

    def test_research_recurrent_constructor_and_dtype(self):
        namespace = definitions(ROOT/'research_scripts/convlstm.py', {'ConvLSTMCell','ConvLSTM'}, {'torch':torch,'nn':nn})
        model = namespace['ConvLSTM'](3,4,(3,3),2).double()
        result,states = model(torch.rand(1,2,3,8,8,dtype=torch.float64))
        self.assertEqual(result[0].dtype,torch.float64)
        self.assertEqual(tuple(result[0].shape),(1,2,4,8,8))

    def test_original_and_packaged_real_case_patches(self):
        import cv2, glob, os, random, re
        from torch.utils.data import Dataset
        ns = {'np':np,'cv2':cv2,'glob':glob,'os':os,'random':random,'re':re,
              'torch':torch,'Dataset':Dataset,'PATCH_SIZE':128,'INPUT_SEQ_LENGTH':5}
        definitions(ROOT/'source_archive/训练代码/单帧预测模型/train_pytorch.py', {'RAMBatteryDataset'},ns)
        case = sorted((ROOT/'examples/data').iterdir())[0]
        with contextlib.redirect_stdout(io.StringIO()):
            original = ns['RAMBatteryDataset']([str(case)],is_training=False)
        actual = BatteryDataset([case],augment=False)
        self.assertEqual(len(original),len(actual))
        for index in (0,3,15,16,len(original)-1):
            a,b = original[index]; x,y = actual[index]
            self.assertTrue(torch.equal(a,x))
            self.assertTrue(torch.equal(b,y[0]))

    def test_stress_only_loader_uses_stress_frames(self):
        specification = importlib.util.spec_from_file_location('original_research_data_loader', ROOT/'research_scripts/data_loader.py')
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp)/'中文路径'
            for folder in ('2_Stress','3_Voronoi_Geometry','C-rate'):
                (case/folder).mkdir(parents=True)
            for path in (case/'2_Stress/stress_t00000.png',case/'2_Stress/stress_t00100.png',case/'3_Voronoi_Geometry/05_orientation.png',case/'C-rate/5C.png'):
                Image.fromarray(np.full((16,16,3),100,np.uint8)).save(path)
            concentration,stress = module.load_simulation_data(str(case),'stress')
            self.assertIsNone(concentration)
            self.assertEqual(stress.shape,(2,512,512,9))
            with self.assertRaises(ValueError):
                module.load_simulation_data(str(case),'unknown')

    def test_empty_export_is_rejected_without_fabricating_data(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp)/'concentration_t01800.png'
            empty.touch()
            with self.assertRaisesRegex(ValueError, 'Empty source image'):
                load_rgb(empty)


if __name__ == '__main__':
    unittest.main()

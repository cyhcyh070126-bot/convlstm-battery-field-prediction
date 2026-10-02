# Reproducibility and source organization

## Selected implementation

This release was prepared from the project's archived MSE + SSIM implementation:

| Original relative filename | Packaged role |
| --- | --- |
| `训练代码/单帧预测模型/train_pytorch.py` | Single-frame dataset, model, objective, and training defaults |
| `训练代码/连续预测模型/finetune_sequence.py` | Multi-step fine-tuning and scheduled sampling |
| `convlstm.py` | Recurrent cells and stacked ConvLSTM |
| `data_loader.py` | Original concentration/stress image-directory conventions |
| `main_D21origin.m` and its geometry helpers | MATLAB microstructure generation |
| `build_full_comsol_model.m` | COMSOL model construction |
| `run_single_simulation.m` | Simulation orchestration and image export |

[`source-manifest.json`](source-manifest.json) records the SHA-256 hashes of
the selected original source files. Checkpoint identity is documented separately
in [checkpoint-provenance.json](checkpoint-provenance.json).

## Model and objective

The forecasting model has **636,515 parameters**:

1. A `Conv3d` maps nine RGB channels to 32 features, with kernel `(3,5,5)`
   and padding `(1,2,2)`.
2. Three ConvLSTM layers use 32 hidden channels and `(5,5)` recurrent kernels.
3. The final hidden frame passes through a `1x1 Conv2d` to three output channels
   and a sigmoid.

Each call starts with zero hidden/cell states, as in the selected source.
Autoregression shifts the five-frame input window, appending the predicted
RGB field and the same six static channels. Recurrent states are initialized
for each model call.

The loss is `MSE + 0.05 * (1 - SSIM)`, using `pytorch_msssim.ssim` with
`data_range=1.0` on RGB images. Multi-step training averages this objective
over future frames. Predictions fed back during training remain connected
to the computation graph.

| Default | Single-frame | Sequence fine-tuning |
| --- | --- | --- |
| Input frames | 5 | 5 |
| Future frames | 1 | 10 |
| Epochs | 100 | 50 |
| Train batch size | 64 | 16 |
| Validation batch size | 64 | 4 |
| Adam learning rate | `1e-3` | `1e-5` |
| Image size | 512 | 512 |
| Training patch | 128 | 128 |
| Validation region | 128-pixel patches | Full image |
| Gradient norm clipping | 5 | 5 |
| Teacher-forcing probability | Not needed | `max(0, 1 - step * 1e-5)` |

The augmentations are horizontal/vertical flips and a 90-degree rotation
applied jointly to the RGB sequence and static conditioning images.

## Implementation and data handling

- A shared model implementation preserves the selected source's parameter
  names and recurrent computations.
- Command-line arguments replace local paths and fixed device selection.
  The archived scripts train concentration images. `--field stress` exposes
  the original data loader's stress-image route for separately trained models.
- Frame loading is lazy rather than keeping the entire collection in RAM.
  Images are sorted numerically, and malformed images or ambiguous static
  inputs raise an error rather than producing a silent black image.
  Static maps use an eight-case LRU cache per dataset instance per worker
  (about 48 MiB at 512 pixels). Training and validation have separate caches.
  Patch training crops before stacking the temporal tensor; tests confirm
  identical patch values for every field and static channel.
- SSIM failures are reported instead of silently dropping that loss term.
- Each run saves its configuration, case split, history, and checkpoint
  metadata. A batch-limited run is explicitly identifiable in its metadata.
- The original training stages independently used 80% and 90% case splits.
  Packaged sequence fine-tuning inherits the pretrained model's saved split
  so that validation cases do not move into training between these stages.
- Evaluation writes labeled PNG, PDF, GIF, and image-metric files for the
  selected case and checkpoint.
- Training and evaluation reject nonempty output directories. Nonfinite
  learning-rate, SSIM-weight, and sampling-decay settings are rejected.
- MATLAB checks the LiveLink connection before packing, uses a unique COMSOL
  model tag, stops on packing nonconvergence, and checks all required image
  exports before reporting success, using the original model and convergence
  thresholds.

## Checkpoints and evaluation

New checkpoints are saved as `best_model.pt` dictionaries containing weights,
configuration, the train/validation case names, epoch, and validation loss.
Tensor-only legacy state dictionaries can also be read. Loading is strict
and uses PyTorch's `weights_only=True`.
Loading a checkpoint for training starts a new fine-tuning run with a freshly
initialized optimizer, random state, and sampling-step counter.

For legacy weights, reuse the pretraining split through `--split-file` when
available. Field checks ensure that weights match the requested target. The
supplied tensor-only checkpoint is recognized as concentration by its SHA-256,
including after renaming. Other legacy weights without field metadata prompt
the user to confirm the target field.

The release includes `checkpoints/mse-ssim-pretrained.pth`, a tensor-only
checkpoint matching the Conv3d + three-layer ConvLSTM implementation. The
source filename, matching MSE + SSIM code, SHA-256, and loading/parity checks
are recorded in [checkpoint-provenance.json](checkpoint-provenance.json).

The README's 5C result is a new 512-pixel, ten-step autoregressive evaluation
of that checkpoint on case 93203. The model consumes frames at 0–400 seconds
and predicts 500–1400 seconds. Average RGB MSE is 0.00271848 and SSIM is
0.968459. These are case-demonstration metrics over normalized RGB images,
including background pixels. The pretrained example targets concentration;
the stress sequences provide simulation references for separate stress training.
The provenance record describes the evaluation scope and checkpoint metadata.

## Verification records

Checked on 2026-10-02:

- Python 3.12.12, PyTorch 2.10.0, OpenCV 4.11.0, and pytorch-msssim 1.0.0.
- Model parity: the original clean `PaperModel` and the packaged model used
  identical state dictionaries and a common seeded input. All state keys
  matched; the maximum output difference was zero (`torch.equal` passed).
- Nine functional tests passed: recurrent rollout/backpropagation; dataset
  fields, ordering, decoding, and static inputs; exact patch values and bounded
  cache behavior; disjoint persisted case splits; checkpoint compatibility and
  field checks; protected output directories; and finite loss/training settings.
- Two real example cases completed a small, one-batch single-frame run,
  ten-step sequence fine-tuning, and ten-step autoregressive evaluation,
  producing PNG/PDF/GIF/CSV/JSON outputs. A separate small stress-route run
  also passed. These used reduced image sizes for execution checks only.
- The README's 32-pixel CPU quick-start command was run successfully.
- The included checkpoint loaded strictly, matched the source model's
  forward output exactly, and completed the full-resolution 5C rollout.
  Its comparison PNG/PDF, GIF, metrics CSV, and JSON report were inspected.
- All 104 example PNG copies were checked against their source SHA-256
  hashes. The example manifest records these hashes and image metadata.
- MATLAB R2025b checked all 13 `.m` files (including tests) without detected
  syntax errors; small contact, force, and circular-boundary helper checks passed.
  Six export-contract regression tests passed, as did the bundled real 5C case's
  52 required PNG checks. These MATLAB checks cover syntax, numerical helpers,
  and the image-export contract.

### Independent CPU installation audit

A new GitHub clone and an isolated virtual environment were created on
2026-10-02, with `include-system-site-packages = false`. PyTorch was installed
from its CPU wheel index, followed by this repository's `requirements.txt`.
`pip check` passed. The verified platform was Windows on an Intel Core i9-13900H.

| Dependency | Installed version |
| --- | --- |
| Python | 3.12.12 |
| PyTorch | 2.10.0+cpu |
| NumPy | 2.5.3 |
| OpenCV headless | 4.14.0.94 |
| pytorch-msssim | 1.0.0 |
| Matplotlib | 3.11.2 |
| Pillow | 12.3.0 |

The independent environment produced the following execution results:

| Command path | Outcome | Approximate elapsed time |
| --- | --- | ---: |
| Functional regression suite | 9 tests passed | 9.0 s including first imports |
| Supplied weights, 5C, 64-pixel/3-step CPU preview | PNG/PDF/GIF/CSV/JSON created | 3.2 s |
| Initial CPU training, 32 pixels, 1 train/validation batch | Checkpoint, split, config, and history saved | 3.5 s |
| Sequence CPU training, 32 pixels, 10 future steps, 1 batch | Saved split inherited and checkpoint saved | 4.1 s |
| Reload sequence checkpoint and predict 10 steps | Prediction files and smoke-marked JSON created | 3.4 s |

Timings describe the listed execution checks on this machine, including process
startup and excluding installation. The matching commands and output names are
described in [getting started](getting-started.md). Generated runs are stored
locally under `outputs/`.

# ConvLSTM Forecasting of Battery Material Fields

**Yanghao Chen · Tongji University**

[Research homepage](https://cyhcyh070126-bot.github.io/) · [MATLAB workflow](docs/matlab-workflow.md) · [Data format](docs/data-format.md) · [Reproducibility](docs/reproducibility.md)

MATLAB–COMSOL simulation and conditional ConvLSTM forecasting of concentration
and stress images in polycrystalline battery materials. This release includes
the **Conv3d + three-layer ConvLSTM model**, its **MSE + SSIM pretrained weights**,
and a complete **5C concentration-prediction example**.

## 1. Dataset and the 5C example

MATLAB generates particle packing and Voronoi grain geometry. COMSOL solves the
coupled transport and solid-mechanics problem and exports concentration and
von Mises stress as RGB image sequences. Each simulation case also supplies a
static grain-orientation map and a static C-rate image. Python combines these
images into conditional forecasting sequences.

The local simulation archive contains **262 case directories**, covering
0.5C, 1C, 2C, 3C, 4C, and 5C. Of these, 261 contain 25 concentration and
25 stress frames; one case is incomplete. This is an archive inventory, not a
reconstruction of the pretrained model's training split. The repository bundles
two complete cases; it does not include the entire image archive.

| C-rate | 0.5C | 1C | 2C | 3C | 4C | 5C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived cases | 40 | 43 | 43 | 44 | 43 | 49 |

The example throughout this README is **5C, case 93203**:
`N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203`.
It contains 60 grains and 512 × 512 RGB exports at 100-second intervals from
0 to 2400 seconds. See the [dataset inventory](docs/dataset-inventory.json)
and [image-format documentation](docs/data-format.md).

| Grain-orientation input | C-rate input: 5C |
| :---: | :---: |
| ![5C case grain orientation](assets/figures/orientation-input.png) | ![5C conditioning image](assets/figures/c-rate-5c.png) |

The C-rate image is spatially uniform: this case encodes 5C as RGB `(153, 0, 0)`.
Its three channels are carried alongside the three orientation channels at
every time step. This is the dataset's image encoding, not an extra predicted
physical field.

| Concentration reference at 1000 s | Stress reference at 1000 s |
| :---: | :---: |
| ![5C concentration reference](assets/figures/concentration-reference.png) | ![5C stress reference](assets/figures/stress-reference.png) |

The orientation and field colors encode different quantities. Historical
exports have different plotting margins and have not been geometrically
registered in this release. The model operates on those rendered images.

## 2. Model and ConvLSTM cell

![ConvLSTM architecture illustration used on the research homepage](assets/figures/battery_convlstm_pipeline.png)

This existing conceptual illustration is reused from the research homepage.
It shows conditioning by previous fields, orientation, and C-rate. The
implemented model includes the additional Conv3d feature extractor listed below.
Illustration source: Wang et al., Figure 5, [Energy Storage Materials 82 (2025),
104581](https://doi.org/10.1016/j.ensm.2025.104581).

Each input contains **five frames**, with nine channels per frame:
three field channels, three grain-orientation channels, and three C-rate
channels. Pixel values are normalized to `[0, 1]`.

| Component | Implemented configuration |
| --- | --- |
| Input | `(batch, 5, 9, height, width)` |
| Conv3d | 9 → 32 channels; kernel `(3, 5, 5)`; padding `(1, 2, 2)` |
| ConvLSTM | Three layers; 32 hidden channels each; 5 × 5 recurrent kernels |
| Output projection | 1 × 1 Conv2d, 32 → 3 channels |
| Output activation | Sigmoid |
| Trainable parameters | 636,515 |

Conv3d extracts local temporal and spatial features before recurrent processing.
The final hidden frame is decoded into the next RGB field. During a rollout,
that prediction is appended to the five-frame window with the same static
conditioning maps. Each model call initializes its recurrent states as in the
selected source implementation.

![ConvLSTM cell structure used on the research homepage](assets/figures/battery_convlstm_cell_clean.png)

The cell's convolutional gates update spatial hidden and cell states.
Illustration source: Wang et al., Figure 4, [same article](https://doi.org/10.1016/j.ensm.2025.104581).
These literature illustrations are explanatory material, not experimental
results or a claim of authorship of the cited paper.

## 3. Training strategy

The code provides initial next-frame training and subsequent multi-step
fine-tuning. The first stage learns local prediction from five previous frames.
The sequence stage unrolls ten future steps and gradually replaces ground-truth
feedback with the model's own predictions through scheduled sampling.

![Scheduled sampling illustration used on the research homepage](assets/figures/battery_scheduled_sampling.png)

Existing illustration from the homepage, corresponding to Figure 6 in
[Wang et al.](https://doi.org/10.1016/j.ensm.2025.104581). The token/softmax labels
are part of the general illustration; this implementation regresses continuous
RGB images and uses a sigmoid output, not token sampling.

| Setting | Initial training | Sequence fine-tuning |
| --- | --- | --- |
| Past / future frames | 5 / 1 | 5 / 10 |
| Image resolution | 512 × 512 | 512 × 512 |
| Training patches | 128 × 128 | 128 × 128 |
| Validation region | 128 × 128 patches | Full images |
| Batch size | 64 | 16 |
| Epochs | 100 | 50 |
| Adam learning rate | `1e-3` | `1e-5` |
| Gradient norm clipping | 5 | 5 |

Training uses joint spatial augmentation of field and static channels.
Sequence teacher-forcing probability is `max(0, 1 - step * 1e-5)`.
Case splitting occurs before temporal windows are formed. Packaged checkpoints
retain their case split, and fine-tuning inherits it to avoid moving validation
cases into training between stages. The historical tensor-only checkpoint does
not retain its original split; its demonstration below is not labeled a held-out
test benchmark.

Concentration and stress are separate prediction targets. The included weights
and results on this page are for **concentration**. The stress images illustrate
the available dataset field; stress prediction requires its own trained model.

## 4. MSE + SSIM objective

Both stages use the same supervised objective:

$$\mathcal{L}=\operatorname{MSE}(\hat I,I)+0.05\,[1-\operatorname{SSIM}(\hat I,I)].$$

MSE measures pixelwise differences, while SSIM compares image structure.
Both terms are computed on normalized RGB images. Multi-step fine-tuning
averages the objective over the predicted future frames. This release contains
the MSE + SSIM implementation; experimental physical-loss training variants are
excluded.

## 5. Pretrained weights and 5C prediction results

[Download the MSE + SSIM pretrained checkpoint](checkpoints/mse-ssim-pretrained.pth).
The weights load strictly into the Conv3d + three-layer ConvLSTM architecture
above. [Checkpoint provenance](docs/checkpoint-provenance.json) records the
source identity, SHA-256, matching code, and verification scope.

This example was evaluated at **512 × 512 resolution**. The model receives
reference frames from **0–400 s** and autoregressively predicts **500–1400 s**,
with no future ground-truth frames fed back during the rollout.

![5C concentration: simulation reference and autoregressive prediction](assets/results/5c/rollout.gif)

![5C concentration prediction and absolute RGB error](assets/results/5c/comparison.png)

[Open the comparison PDF](assets/results/5c/comparison.pdf) ·
[Per-frame metrics CSV](assets/results/5c/metrics.csv) ·
[Evaluation settings and summary](assets/results/5c/evaluation.json)

| RGB image metric | Mean over 10 predicted frames |
| --- | ---: |
| MSE | 0.00271848 |
| SSIM | 0.968459 |

These values were computed from this checkpoint and this 5C case during repository
preparation. They include background pixels. They quantify **rendered-image
agreement**, not concentration error in mol/m³. The case's original training-set
membership is unknown; this is a reproducible case demonstration. Prediction
error accumulates through the rollout, visible in the per-frame CSV and figures.

## 6. Run the example

Use Python 3.10 or newer. Preparation checks used Python 3.12 and PyTorch 2.10.

```bash
git clone https://github.com/cyhcyh070126-bot/convlstm-battery-field-prediction.git
cd convlstm-battery-field-prediction
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Install the PyTorch build appropriate
for your hardware, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the included checkpoint on the 5C case:

```bash
python -m python.evaluation.predict --case-dir "examples/data/N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203" --checkpoint checkpoints/mse-ssim-pretrained.pth --output-dir outputs/5c-prediction --field concentration --predict-length 10
```

Use a new or empty output directory. The evaluator writes a comparison PNG/PDF,
GIF, predicted PNG frames, per-frame CSV metrics, and a JSON report.

## 7. Train or generate new data

For a short CPU execution check with the bundled cases:

```bash
python -m python.train.single_frame --data-dir examples/data --output-dir outputs/smoke --field concentration --epochs 1 --batch-size 1 --val-batch-size 1 --image-size 32 --patch-size 32 --max-batches 1 --device cpu
```

For full training, arrange your own cases as in the [data-format guide](docs/data-format.md):

```bash
python -m python.train.single_frame --data-dir data/export_images --output-dir outputs/concentration-initial --field concentration
python -m python.train.sequence --data-dir data/export_images --checkpoint outputs/concentration-initial/best_model.pt --output-dir outputs/concentration-sequence --field concentration
```

Use `--field stress` with separate output directories to train a stress model.
The bundled cases are execution examples, not a replacement for a full training
and validation collection. Run `--help` for configurable batch sizes, device,
learning rate, and rollout length.

To generate new simulations, use a MATLAB session connected to COMSOL LiveLink:

```matlab
addpath('matlab');
run_dir = run_workflow(fullfile(pwd, 'outputs', 'matlab'), '', 5, 42);
```

The final arguments select 5C and random seed 42. The [MATLAB guide](docs/matlab-workflow.md)
details dependencies and outputs. MATLAB is not needed to use the bundled images.

## Repository layout and checks

```text
matlab/        Geometry, COMSOL model construction, and image export
python/        Models, image loading, training, and evaluation
checkpoints/   Included MSE + SSIM pretrained weights
examples/      Two complete sample cases and source hashes
assets/        Existing architecture illustrations, 5C inputs, and prediction figures
docs/          Data format, simulation setup, source provenance, and reproducibility
tests/         Functional model/data/checkpoint checks
```

Run `python -m unittest discover -s tests -v`. Model parity, small training runs,
checkpoint loading, and the included 5C prediction have been checked. MATLAB
scripts passed syntax checks and small helper checks. A full COMSOL simulation
and full training campaign were not rerun during repository preparation.
See [reproducibility notes](docs/reproducibility.md) for details.

## References and attribution

- Wang, Z., Zhao, Y., Zhong, Z., and Xu, B.-X. *Deep-learning based prediction of chemo-mechanics and damage in battery active materials*. Energy Storage Materials 82 (2025), 104581. [Article](https://doi.org/10.1016/j.ensm.2025.104581). Figures 4–6 correspond to the existing homepage illustrations reused above; they are not covered by the ConvLSTM code license.
- The recurrent implementation follows [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch); its [MIT notice](third_party/ConvLSTM_pytorch-LICENSE) is retained.
- SSIM uses [pytorch-msssim](https://github.com/VainF/pytorch-msssim).

See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and licensing scope.

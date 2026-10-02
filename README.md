# ConvLSTM Forecasting of Battery Material Fields

**Yanghao Chen · Tongji University**

[Research homepage](https://cyhcyh070126-bot.github.io/) · [MATLAB workflow](docs/matlab-workflow.md) · [Data format](docs/data-format.md) · [Reproducibility](docs/reproducibility.md)

MATLAB–COMSOL simulation and conditional ConvLSTM forecasting of concentration
and stress images in polycrystalline battery materials. This release includes
the **Conv3d + three-layer ConvLSTM model**, its **MSE + SSIM pretrained weights**,
and a complete **5C concentration-prediction example**.

**Start here:** [install and predict on CPU](#6-run-the-example) ·
[check the training pipeline](#7-train-or-generate-new-data) ·
[troubleshooting](docs/getting-started.md#troubleshooting).
The bundled images and weights are enough to run Python prediction immediately;
MATLAB and COMSOL are only needed to generate new simulation data.

## 1. Dataset and the 5C example

MATLAB generates particle packing and Voronoi grain geometry. COMSOL solves the
coupled transport and solid-mechanics problem and exports concentration and
von Mises stress as RGB image sequences. Each simulation case also supplies a
static grain-orientation map and a static C-rate image. Python combines these
images into conditional forecasting sequences.

The simulation archive spans **262 cases** across 0.5C, 1C, 2C, 3C, 4C, and 5C.
The repository provides **two complete example cases**, each with 25 concentration
frames, 25 stress frames, and the associated conditioning maps, ready for
prediction and training walkthroughs. The archive's C-rate distribution is:

| C-rate | 0.5C | 1C | 2C | 3C | 4C | 5C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived cases | 40 | 43 | 43 | 44 | 43 | 49 |

The input maps and pretrained prediction example use **5C, case 93203**:
`N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203`.
It contains 60 grains and 512 × 512 RGB exports at 100-second intervals from
0 to 2400 seconds. See the [dataset inventory](docs/dataset-inventory.json)
and [image-format documentation](docs/data-format.md).

| Grain-orientation input | C-rate input: 5C |
| :---: | :---: |
| ![5C case grain orientation](assets/figures/orientation-input.png) | ![5C conditioning image](assets/figures/c-rate-5c.png) |

The C-rate image is spatially uniform: this case encodes 5C as RGB `(153, 0, 0)`.
Its three channels are carried alongside the three orientation channels at
every time step as a conditioning input.

| Concentration evolution | Von Mises stress evolution |
| :---: | :---: |
| ![Dataset concentration evolution from the research homepage](assets/gifs/dataset-concentration.gif) | ![Dataset von Mises stress evolution from the research homepage](assets/gifs/dataset-stress.gif) |

These dataset-reference animations are reused unchanged from the
[research homepage](https://cyhcyh070126-bot.github.io/cv/). The 5C input maps
above and prediction results in Section 5 identify the specific example case.
The GIFs illustrate dataset evolution; the labeled 5C example pairs its own
conditioning maps with the predictions in Section 5. The model uses rendered
RGB images; see the [image representation](docs/data-format.md#image-representation)
for color conventions and coordinate extents.

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
Figures 4–6 from the cited article illustrate the recurrent model and training concepts.

## 3. Training strategy

The code provides initial next-frame training and subsequent multi-step
fine-tuning. The first stage learns local prediction from five previous frames.
The sequence stage unrolls ten future steps and gradually replaces ground-truth
feedback with the model's own predictions through scheduled sampling.

![Scheduled sampling illustration used on the research homepage](assets/figures/battery_scheduled_sampling.png)

Existing illustration from the homepage, corresponding to Figure 6 in
[Wang et al.](https://doi.org/10.1016/j.ensm.2025.104581). The illustration introduces
the general scheduled-sampling strategy; the model here applies it to continuous
RGB regression with a sigmoid output.

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
retain their case split, and fine-tuning inherits it across the two stages.
Section 5 presents a case study using the supplied pretrained weights.

Concentration and stress use separate models. The supplied pretrained weights
and 5C predictions are for **concentration**. The dataset also includes stress
images for training through `--field stress`.

## 4. MSE + SSIM objective

Both stages use the same supervised objective:

```math
\mathcal{L}=\mathrm{MSE}(\hat{I},I)+0.05\left[1-\mathrm{SSIM}(\hat{I},I)\right]
```

MSE measures pixelwise differences, while SSIM compares image structure.
Both terms are computed on normalized RGB images. Multi-step fine-tuning
averages the objective over the predicted future frames. Both training entry
points use this MSE + SSIM objective.

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

The metrics describe this **5C case study** using the supplied checkpoint.
MSE and SSIM measure agreement over normalized RGB images, including background
pixels. The per-frame CSV and error maps show how image accuracy changes across
the ten-step rollout. See the [evaluation record](docs/checkpoint-provenance.json)
for checkpoint provenance and evaluation scope.

## 6. Run the example

Use **Python 3.12** for the tested installation below. No GPU is needed for the
quick preview. Run all commands from the repository root.

```bash
git clone https://github.com/cyhcyh070126-bot/convlstm-battery-field-prediction.git
cd convlstm-battery-field-prediction
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. If PowerShell blocks activation,
replace `python` in the remaining commands with `.venv\Scripts\python.exe`;
no system policy change is needed. For the tested Windows CPU installation
(the CPU index also provides Linux wheels):

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

For CUDA or macOS, choose the matching build using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/)
before installing `requirements.txt`. The CPU build above cannot use CUDA.

**First result: a small CPU preview.** This uses the supplied MSE + SSIM weights,
resizes the images to 64 × 64, and predicts three steps:

```bash
python -m python.evaluation.predict --case-dir "examples/data/N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203" --checkpoint checkpoints/mse-ssim-pretrained.pth --output-dir outputs/5c-preview --field concentration --image-size 64 --predict-length 3 --device cpu
```

Open `outputs/5c-preview/rollout.gif` or `comparison.png`. Expect three predicted
PNG frames, a comparison PNG/PDF, a GIF, `metrics.csv`, and `evaluation.json`.
This preview took about 3 seconds on the audited i9-13900H laptop after installation;
other machines will differ. This command produces a 64-pixel, three-step preview;
the command below uses the 512-pixel, ten-step configuration of Section 5.

**Reproduce the displayed 5C evaluation** at 512 × 512 and ten steps:

```bash
python -m python.evaluation.predict --case-dir "examples/data/N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203" --checkpoint checkpoints/mse-ssim-pretrained.pth --output-dir outputs/5c-prediction --field concentration --predict-length 10
```

Use a new or empty output directory. The evaluator writes a comparison PNG/PDF,
GIF, predicted PNG frames, per-frame CSV metrics, and a JSON report. The default
device is CUDA when available, otherwise CPU. Full-resolution execution is much
heavier than the preview; published figures were evaluated on a GPU.
See [setup, expected files, and common errors](docs/getting-started.md).

## 7. Train or generate new data

For a short CPU execution check with the bundled cases:

```bash
python -m python.train.single_frame --data-dir examples/data --output-dir outputs/smoke --field concentration --epochs 1 --batch-size 1 --val-batch-size 1 --image-size 32 --patch-size 32 --max-batches 1 --device cpu
```

Success produces `best_model.pt`, `config.json`, `split.json`, and `history.csv`.
This command processes one training batch and one validation batch to check
installation and checkpoint saving. The
[complete CPU walkthrough](docs/getting-started.md#check-both-training-stages)
also tests sequence fine-tuning and reloading its saved weights.

For full training, arrange your own cases as in the [data-format guide](docs/data-format.md):

```bash
python -m python.train.single_frame --data-dir data/export_images --output-dir outputs/concentration-initial --field concentration
python -m python.train.sequence --data-dir data/export_images --checkpoint outputs/concentration-initial/best_model.pt --output-dir outputs/concentration-sequence --field concentration
```

Use `--field stress` with separate output directories to train a stress model.
The bundled cases support the execution walkthrough. For a research experiment,
point `--data-dir` to your case collection and retain the case-level split.
Run `--help` for configurable batch sizes, device, learning rate, and rollout length.

Default batch sizes are 64 for initial training and 16 for sequence training.
For a memory-efficient starting configuration, use `--batch-size 1 --val-batch-size 1`,
especially for ten-step training. Static-image caching is bounded to eight cases
per dataset instance per worker; `--num-workers 0` is the simplest starting point.
`--checkpoint` initializes weights for a new fine-tuning run with a fresh
optimizer and sampling schedule. Use a new or empty output directory for each run.

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

Run `python -m unittest discover -s tests -v`. The nine Python tests cover
model execution, data loading, checkpoints, and output protection. Additional
checks cover the two-stage training walkthrough, 5C prediction, MATLAB syntax,
and six image-export tests. See [reproducibility notes](docs/reproducibility.md)
for environments and execution records.

## References and attribution

- Wang, Z., Zhao, Y., Zhong, Z., and Xu, B.-X. *Deep-learning based prediction of chemo-mechanics and damage in battery active materials*. Energy Storage Materials 82 (2025), 104581. [Article](https://doi.org/10.1016/j.ensm.2025.104581). Figures 4–6 correspond to the existing homepage illustrations reused above; they are not covered by the ConvLSTM code license.
- The recurrent implementation follows [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch); its [MIT notice](third_party/ConvLSTM_pytorch-LICENSE) is retained.
- SSIM uses [pytorch-msssim](https://github.com/VainF/pytorch-msssim).

See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and licensing scope.

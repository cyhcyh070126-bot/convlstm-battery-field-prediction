# ConvLSTM Prediction of Battery Material Fields

**Yanghao Chen · Tongji University**

This repository contains the MATLAB–COMSOL simulation workflow and PyTorch
models used to predict concentration and von Mises stress image sequences in
polycrystalline battery materials. Separate conditional ConvLSTM models use
field histories, grain orientations, and C-rate maps to predict subsequent frames.

[Sample](#what-one-simulation-case-contains) · [Dataset generation](#dataset-and-simulation-workflow) · [Model](#model-architecture) · [Training](#training-strategy) · [5C results](#5c-prediction-results) · [Run the workflow](#quick-start) · [Documentation](#documentation) · [Research homepage](https://cyhcyh070126-bot.github.io/)

The release includes two simulation cases and a concentration checkpoint for
CPU or GPU inference. New simulation cases require MATLAB with COMSOL LiveLink.
Stress images and training code are included; a pretrained stress model is not.

## What one simulation case contains

The example is **5C, case 93203**:
`N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203`.
It was generated from 60 initial packing particles and contains 512 × 512 RGB
field exports at 100-second intervals from 0 to 2400 seconds. See the [dataset inventory](docs/dataset-inventory.json)
and [image-format documentation](docs/data-format.md).

A case comprises one microstructure, its loading condition, and the resulting
field histories. The default export produces the following images:

| Image group | PNGs per case | Role |
| :--- | ---: | :--- |
| Concentration | 25 | Field sequence at 0, 100, …, 2400 s |
| Von Mises stress | 25 | Mechanical response at the same times |
| Grain orientation | 1 | Static microstructure input |
| C-rate | 1 | Static loading-condition input |
| Geometry construction | 4 | Radius distribution, initial/final packing, and Voronoi grains |
| **Total** | **56** | **52 learning images + 4 construction plots** |

The learning images are organized in `1_Concentration/`, `2_Stress/`,
`3_Voronoi_Geometry/05_*.png`, and `C-rate/5C.png` for this case.

The 52 learning images are 512 × 512 RGB. The four original construction plots
retain their 2027 × 1221 resolution. For case 93203, all 56 images are available:
the learning images are in [`examples/data/`](examples/data/), and the construction
plots are in [`assets/sample-5c/`](assets/sample-5c/).
New MATLAB–COMSOL runs also save geometry and run metadata (`.mat`) together
with the solved COMSOL model (`.mph`). The
[illustrated sample guide](docs/simulation-sample.md) explains each file and how
one case becomes many training windows.

### Static conditioning and evolving fields

| Grain-orientation input | C-rate input: 5C |
| :---: | :---: |
| ![5C case grain orientation](assets/figures/orientation-input.png) | ![5C conditioning image](assets/figures/c-rate-5c.png) |
| Each polygon is one grain, colored by its folded crystal orientation. This is the structural input for case 93203. | The uniform dark-red image encodes 5C as RGB `(153, 0, 0)`. It is a loading-condition input, not a concentration or stress field. |

The two static maps supply six conditioning channels at every time step.

The [conditioning guide](docs/simulation-sample.md#additional-orientation-map-example)
also includes the additional grain-orientation image shown on the research homepage.

| Concentration evolution | Von Mises stress evolution |
| :---: | :---: |
| ![Dataset concentration evolution from the research homepage](assets/gifs/dataset-concentration.gif) | ![Dataset von Mises stress evolution from the research homepage](assets/gifs/dataset-stress.gif) |
| Simulation-reference concentration images evolving over physical time. The colors represent the exported concentration field. | Simulation-reference von Mises stress images evolving over physical time. Stress uses a different color encoding from concentration. |

These are simulation-reference animations from the
[research homepage](https://cyhcyh070126-bot.github.io/cv/). Their case IDs and
physical-time labels are not recorded. Labeled snapshots from case 93203 are
provided in the [sample guide](docs/simulation-sample.md#3-field-sequences);
its network predictions appear in [5C prediction results](#5c-prediction-results).

<a id="1-dataset-and-the-5c-example"></a>

## Dataset and simulation workflow

MATLAB and COMSOL generate each case together through LiveLink for MATLAB:

1. **Construct the microstructure.** MATLAB samples particle radii, relaxes the
   particle positions, constructs Voronoi grains, and assigns crystal orientations.
2. **Solve the physical fields.** MATLAB creates and configures the COMSOL model;
   COMSOL solves anisotropic transport and solid mechanics with concentration-driven swelling.
3. **Export the learning data.** MATLAB controls the time-dependent solve and
   image export, pairing concentration and von Mises stress sequences with static
   orientation and C-rate maps. Python then forms prediction windows from these images.

The local simulation archive contains **261 complete cases** across six C-rates.
The public repository includes two of these cases, with both field sequences
and their conditioning maps. The full archive is not bundled. Its case counts are:

| C-rate | 0.5C | 1C | 2C | 3C | 4C | 5C |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archived cases | 40 | 43 | 43 | 43 | 43 | 49 |

### From particles to grains

| Final particle packing, case 93203 | Voronoi grains colored by seed-particle radius |
| :---: | :---: |
| ![Final particle packing for the 5C example](assets/sample-5c/03_Final_Placement.png) | ![Voronoi microstructure colored by seed-particle radius for the 5C example](assets/sample-5c/04_Voronoi_Uncolored.png) |
| Relaxed seed-particle discs inside the circular packing boundary. Disc colors distinguish particles; they do not encode crystal orientation. | Voronoi grain outlines colored by their generating particle's radius. The colorbar spans 1.2–2.6 µm; this is a radius map, not the orientation input above. |

The [construction gallery](docs/simulation-sample.md#1-microstructure-construction)
also includes the sampled radius distribution and initial particle placement.

### Generate new MATLAB–COMSOL cases

Use a MATLAB session connected to COMSOL LiveLink:

```matlab
addpath('matlab');
run_dir = run_workflow(fullfile(pwd, 'outputs', 'matlab'), '', 5, 42);
```

The final arguments select 5C and random seed 42. The [MATLAB guide](docs/matlab-workflow.md)
details dependencies and outputs. MATLAB is not needed to use the bundled images.

### Choose particle count and radius distribution

Set the geometry options before calling `run_workflow`; these select the
existing branches in [`main_D21origin.m`](matlab/geometry/main_D21origin.m).

| Parameter | What to change | Saved-source default | Worked 5C case 93203 |
| :--- | :--- | :--- | :--- |
| `N` | Number of initial packing particles / Voronoi seeds; an integer ≥ 3 | `60` | `60` |
| `command` | Radius distribution: `1` = Weibull, `2` = Lognormal, `3` = Normal | `3` (Normal) | `2` (Lognormal) |
| `mu` | Radius mean parameter in µm; positive | `2` | `2` |
| `sigma` | Radius standard-deviation parameter in µm; positive | `0.35` | `0.10` |

For example, generate a new **80-particle Lognormal** case:

```matlab
addpath('matlab');
geometry_options = struct();
geometry_options.N = 80;        % Change to 40, 60, 80, ...
geometry_options.command = 2;   % 1: Weibull; 2: Lognormal; 3: Normal
geometry_options.mu = 2.00;     % Radius mean parameter, micrometres
geometry_options.sigma = 0.15;  % Radius standard deviation, micrometres

c_rate = 5;                     % Loading condition: 5C
random_seed = 42;               % Use [] for randomly seeded geometry
run_dir = run_workflow(fullfile(pwd, 'outputs', 'matlab'), ...
    '', c_rate, random_seed, geometry_options);
```

For Lognormal radii, `mu` and `sigma` are the radius-space mean and standard
deviation, converted internally to log-space parameters. For Weibull radii,
the code fits shape and scale from these two moments. All three distributions
use the original binning and sample-count adjustment, so the finite sample's
statistics need not equal the prescribed moments exactly.

`N` counts the initial seeds, so the final number of polygons after boundary
clipping can differ. The initial `R0` is calculated from the sampled radii and
the original packing fraction, and the packing loop can expand it to reduce
overlap; it is not an independent `geometry_options` field. Change
`c_rate` for another loading condition and `random_seed` for another random
realization. These settings generate a new case; they do not recreate archived
case 93203, whose original seed is not supplied.

Omitted geometry fields retain their saved-source defaults. The
[MATLAB guide](docs/matlab-workflow.md#original-settings) describes the full
setup and output files.

### From simulated cases to network samples

Training examples are overlapping history/target windows within each case.
Each history frame is paired with the same spatially aligned orientation and
C-rate maps.

| Training stage | History → target frames | Windows per 25-frame case | Training patches per case |
| :--- | :---: | ---: | ---: |
| Initial next-frame training | 5 → 1 | 20 | 320 |
| Sequence fine-tuning | 5 → 10 | 11 | 176 |

A 512 × 512 image supplies sixteen non-overlapping 128 × 128 patches.
Sequence validation uses the 11 full-image windows. See
[window construction](docs/simulation-sample.md#4-from-one-case-to-training-examples)
for the tensor shapes.

Case splitting occurs before temporal windows are formed. Checkpoints saved by
the training commands retain their case split, and fine-tuning inherits it
across the two stages. This keeps overlapping windows from the same simulation
out of different training and validation partitions.

Files passed between stages:

| File or folder | Produced by | Used next for |
| :--- | :--- | :--- |
| `geometry_for_comsol.mat` | Particle packing and Voronoi construction | COMSOL geometry and crystal-orientation assignment |
| Solved `.mph` model | Coupled transient COMSOL solve | Concentration and stress image export |
| `1_Concentration/`, `2_Stress/` | Time-aligned field export | Separate concentration or stress prediction windows |
| `3_Voronoi_Geometry/05_*.png`, `C-rate/*.png` | Static conditioning export | Six conditioning channels alongside each RGB field frame |
| `run_metadata.mat` | MATLAB workflow | Geometry options, seed, loading and packing diagnostics |
| `split.json`, saved checkpoint | Python training | Case partitions, model weights and subsequent fine-tuning/evaluation |

<a id="2-model-and-convlstm-cell"></a>

## Model architecture

![ConvLSTM architecture illustration used on the research homepage](assets/figures/battery_convlstm_pipeline.png)

The schematic shows field and conditioning inputs, recurrent processing, and
RGB decoding. This implementation also includes a Conv3d feature extractor
before the recurrent layers, which is absent from the schematic. Source:
Wang et al., Figure 5, [Energy Storage Materials 82 (2025),
104581](https://doi.org/10.1016/j.ensm.2025.104581).

Each training example takes a **five-frame history**, with nine channels per frame:
three field channels, three grain-orientation channels, and three C-rate
channels. Pixel values are normalized to `[0, 1]`.

For example, the first concentration window uses **0, 100, 200, 300, and 400 s**.
The initial training stage targets **500 s**; sequence fine-tuning targets
**500–1400 s**. The two static maps accompany every history frame, and the
target contains only the three RGB channels of the selected field.

| Component | Implemented configuration |
| --- | --- |
| Input | `(batch, 5, 9, height, width)` |
| Conv3d | 9 → 32 channels; kernel `(3, 5, 5)`; padding `(1, 2, 2)` |
| ConvLSTM | Three layers; 32 hidden channels each; 5 × 5 recurrent kernels |
| Output projection | 1 × 1 Conv2d, 32 → 3 channels |
| Output activation | Sigmoid |
| Trainable parameters | 636,515 |

Conv3d extracts local temporal and spatial features before recurrent processing.
The hidden state at the final input time is decoded into the next RGB field. During a rollout,
that prediction is appended to the five-frame window with the same static
conditioning maps. Each model call initializes its recurrent states as in the
selected source implementation.

![ConvLSTM cell structure used on the research homepage](assets/figures/battery_convlstm_cell_clean.png)

Within each ConvLSTM cell, convolutional gates update the hidden state $H_t$
and cell state $C_t$ from the input features $X_t$. The diagram depicts one
cell; the model stacks three layers. Source: Wang et al., Figure 4,
[same article](https://doi.org/10.1016/j.ensm.2025.104581).

<a id="3-training-strategy"></a>

## Training strategy

The code provides initial next-frame training and subsequent multi-step
fine-tuning. The first stage learns local prediction from five previous frames.
The sequence stage unrolls ten future steps and gradually replaces ground-truth
feedback with the model's own predictions through scheduled sampling.

![Scheduled sampling illustration used on the research homepage](assets/figures/battery_scheduled_sampling.png)

Scheduled sampling feeds back the simulation reference with probability
$\epsilon$ and the prediction with probability $1-\epsilon$. This RGB regression
model uses sigmoid outputs directly; the schematic's Softmax and Sample blocks
describe a discrete-output example. Source: Wang et al., Figure 6,
[same article](https://doi.org/10.1016/j.ensm.2025.104581).

| Setting | Initial training | Sequence fine-tuning |
| :--- | :---: | :---: |
| History → target frames | 5 → 1 | 5 → 10 |
| Training patch | 128 × 128 | 128 × 128 |
| Validation region | 128 × 128 patches | 512 × 512 images |
| Batch size | 64 | 16 |
| Epochs | 100 | 50 |
| Adam learning rate | `1e-3` | `1e-5` |

Both stages use 512 × 512 source images, joint spatial augmentation of field
and static channels, and gradient-norm clipping at 5. The sequence-stage
teacher-forcing probability is `max(0, 1 - step * 1e-5)`, where `step` counts
optimizer updates. Use `--field stress` to train the separate stress model.

These settings describe the training code. The supplied legacy checkpoint
contains weights only, so its training epoch, case split, and use of sequence
fine-tuning cannot be recovered from the file.

<a id="4-mse--ssim-objective"></a>

### MSE + SSIM objective

Both stages use the same supervised objective:

```math
\mathcal{L}=\mathrm{MSE}(\hat{I},I)+0.05\left[1-\mathrm{SSIM}(\hat{I},I)\right]
```

MSE measures pixelwise differences, while SSIM compares image structure.
Both terms are computed on normalized RGB images. Multi-step fine-tuning
averages this objective over the predicted future frames.

<a id="5-pretrained-weights-and-5c-prediction-results"></a>

## 5C prediction results

The 5C example uses five reference frames from **0–400 s** to predict ten
future frames at **500–1400 s**, at 512 × 512 resolution. Each predicted frame
is fed back into the input window; future reference frames are used for evaluation.

[Pretrained checkpoint](checkpoints/mse-ssim-pretrained.pth) · [Checkpoint provenance](docs/checkpoint-provenance.json)

![5C concentration: simulation reference and autoregressive prediction](assets/results/5c/rollout.gif)

**Concentration rollout, case 93203.** Left: COMSOL reference. Right: prediction
from the fixed pretrained model. The ten frames span 500–1400 s in 100 s steps;
`concentration_t01000` denotes 1000 s.

![5C concentration prediction and absolute RGB error](assets/results/5c/comparison.png)

**Selected rollout frames at 500, 800, 1100, and 1400 s.** Rows show the COMSOL
reference, prediction, and mean absolute error over the three normalized RGB
channels. The colorbar reports image-space error, not concentration units.

[Open the comparison PDF](assets/results/5c/comparison.pdf) ·
[Per-frame metrics CSV](assets/results/5c/metrics.csv) ·
[Evaluation settings and summary](assets/results/5c/evaluation.json)

| Metric | Mean over 10 frames |
| :--- | ---: |
| RGB MSE | 0.00271848 |
| RGB SSIM | 0.968459 |

Over the rollout, RGB MSE increases from `2.89e-5` at 500 s to `8.08e-3`
at 1400 s, while SSIM decreases from `0.9991` to `0.9258`. The per-frame
record therefore shows the accumulated prediction error that the mean values summarize.

Metrics include background pixels. The historical training split is unavailable,
so the held-out status of this case is unknown; this is a reproducible case
evaluation, not a generalization benchmark. Checkpoint identity, missing metadata,
and evaluation settings are recorded in
[checkpoint provenance](docs/checkpoint-provenance.json).

<a id="6-run-the-example"></a>

## Quick start

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

### Run a CPU preview

This uses the supplied MSE + SSIM weights,
resizes the images to 64 × 64, and predicts three steps:

```bash
python -m python.evaluation.predict --case-dir "examples/data/N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203" --checkpoint checkpoints/mse-ssim-pretrained.pth --output-dir outputs/5c-preview --field concentration --image-size 64 --predict-length 3 --device cpu
```

Open `outputs/5c-preview/rollout.gif` or `comparison.png`. Expect three predicted
PNG frames, a comparison PNG/PDF, a GIF, `metrics.csv`, and `evaluation.json`.
The preview uses a smaller spatial grid and shorter rollout. To reproduce the
configuration of the [displayed results](#5c-prediction-results), use the next command.

### Reproduce the 5C evaluation

Use the full 512 × 512 images and predict ten steps:

```bash
python -m python.evaluation.predict --case-dir "examples/data/N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203" --checkpoint checkpoints/mse-ssim-pretrained.pth --output-dir outputs/5c-prediction --field concentration --predict-length 10
```

Use a new or empty output directory. The evaluator writes a comparison PNG/PDF,
GIF, predicted PNG frames, per-frame CSV metrics, and a JSON report. The default
device is CUDA when available, otherwise CPU. Full-resolution execution is much
heavier than the preview; published figures were evaluated on a GPU.
See [setup, expected files, and common errors](docs/getting-started.md).

<a id="7-train-or-generate-new-data"></a>

<a id="train-and-generate-data"></a>

## Train on the dataset

### Check the training pipeline

For a short CPU execution check with the bundled cases:

```bash
python -m python.train.single_frame --data-dir examples/data --output-dir outputs/smoke --field concentration --epochs 1 --batch-size 1 --val-batch-size 1 --image-size 32 --patch-size 32 --max-batches 1 --device cpu
```

Success produces `best_model.pt`, `config.json`, `split.json`, and `history.csv`.
This command processes one training batch and one validation batch to check
installation and checkpoint saving. The
[complete CPU walkthrough](docs/getting-started.md#check-both-training-stages)
also tests sequence fine-tuning and reloading its saved weights.

### Train on a case collection

Arrange your cases as in the [data-format guide](docs/data-format.md), then run the two stages in order:

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

## Documentation

The [source audit](docs/source-audit.md) records the supplied MATLAB/Python
files, their roles, and changes made for execution. `source_archive/` preserves
the originals; `research_scripts/` contains the reviewed historical versions.
The commands above use the packaged implementation in `matlab/` and `python/`.

| Guide | What it covers |
| :--- | :--- |
| [Getting started](docs/getting-started.md) | Installation, expected outputs, and troubleshooting |
| [A complete simulation sample](docs/simulation-sample.md) | Microstructure, field images, conditioning maps, and training windows |
| [MATLAB workflow](docs/matlab-workflow.md) | Geometry generation, COMSOL setup, and image export |
| [Data format](docs/data-format.md) | Folder layout and image conventions |
| [Reproducibility](docs/reproducibility.md) | Verified environments, execution records, and checks |

For common setup problems, see [troubleshooting](docs/getting-started.md#troubleshooting).

## Repository layout and checks

```text
matlab/        Geometry, COMSOL model construction, and image export
python/        Models, image loading, training, and evaluation
source_archive/ Exact original source snapshots, for provenance
research_scripts/ Reviewed original versions, with explicit patch records
checkpoints/   Included MSE + SSIM pretrained weights
examples/      Two complete sample cases and source hashes
assets/        Existing architecture illustrations, 5C inputs, and prediction figures
docs/          Data format, simulation setup, source provenance, and reproducibility
tests/         Functional model/data/checkpoint checks
```

Run `python -m unittest discover -s tests -v`. The 15 Python tests cover
model execution, data loading, checkpoints, output protection and original-source
numerical parity. Additional
checks cover the two-stage training walkthrough, 5C prediction, MATLAB syntax,
and six image-export tests. See [reproducibility notes](docs/reproducibility.md)
for environments and execution records.

## References and attribution

- Wang, Z., Zhao, Y., Zhong, Z., and Xu, B.-X. *Deep-learning based prediction of chemo-mechanics and damage in battery active materials*. Energy Storage Materials 82 (2025), 104581. [Article](https://doi.org/10.1016/j.ensm.2025.104581). Figures 4–6 correspond to the existing homepage illustrations reused above; they are not covered by the ConvLSTM code license.
- The recurrent implementation follows [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch); its [MIT notice](third_party/ConvLSTM_pytorch-LICENSE) is retained.
- SSIM uses [pytorch-msssim](https://github.com/VainF/pytorch-msssim).

See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and licensing scope.

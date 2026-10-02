# Running the repository

Start with the [README installation and CPU preview](../README.md#6-run-the-example).
Use Python 3.12 and execute modules from the repository root. You do not need
MATLAB, COMSOL, the full simulation archive, or a GPU for this preview.

## What is included

| Goal | Entry point |
| --- | --- |
| Run the supplied concentration predictor | Pretrained weights and two complete image cases |
| Inspect the 5C prediction without installing anything | `assets/results/5c/` |
| Exercise both training stages | CPU training walkthrough below |
| Train a separate stress predictor | Stress images and the `--field stress` training option |
| Run a training experiment | Your simulation case collection, the training CLI, and saved case-level splits |
| Generate a new simulation | MATLAB and COMSOL LiveLink workflow; see the [MATLAB guide](matlab-workflow.md) for setup |

## Check both training stages

After installation, run these commands in order. Use fresh output directory
names when repeating them. They keep the actual model and MSE + SSIM loss but
reduce resolution, batch count, and epoch count for an execution check.

```bash
python -m python.train.single_frame --data-dir examples/data --output-dir outputs/smoke-initial --field concentration --epochs 1 --batch-size 1 --val-batch-size 1 --image-size 32 --patch-size 32 --max-batches 1 --device cpu
python -m python.train.sequence --data-dir examples/data --checkpoint outputs/smoke-initial/best_model.pt --output-dir outputs/smoke-sequence --field concentration --epochs 1 --batch-size 1 --val-batch-size 1 --image-size 32 --patch-size 32 --max-batches 1 --device cpu
python -m python.evaluation.predict --case-dir "examples/data/N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203" --checkpoint outputs/smoke-sequence/best_model.pt --output-dir outputs/smoke-reloaded --field concentration --predict-length 10 --device cpu
```

The two training stages each produce:

```text
best_model.pt   Weights, configuration, split, epoch, and validation loss
config.json     Actual CLI settings, including the batch limit
split.json      Disjoint train/validation case names
history.csv    Epoch losses and teacher-forcing probability
```

The last command reads the saved 32-pixel configuration and writes ten predicted
frames plus `comparison.png`, `comparison.pdf`, `rollout.gif`, `metrics.csv`, and
`evaluation.json`. The JSON identifies that the checkpoint came from a
batch-limited smoke run. To view the pretrained 5C predictions, use the supplied
MSE + SSIM weights with the README's prediction command.

In a fresh CPU environment on the audited i9-13900H machine, the three commands
took approximately 3.5, 4.1, and 3.4 seconds respectively, including Python
startup. These timings describe the CPU walkthrough after installation.
Details and versions are in [reproducibility notes](reproducibility.md).

## Moving to your own data

Use one subdirectory per independent simulation case, following the
[data-format contract](data-format.md). At least two complete cases are needed
for a disjoint train/validation split. Initial training needs at least six
frames per case; the default ten-step sequence stage needs at least fifteen.
Use independent simulation cases to evaluate performance across microstructures
and C-rates.

Keep concentration and stress runs separate. The supplied checkpoint is verified
as concentration by its SHA-256, including after renaming, and rejects a stress
request. Newly saved checkpoints also record their target field. Unknown legacy
weights without field metadata prompt you to confirm their training target
and original data split.

`--checkpoint` loads weights into a **new** fine-tuning run. Adam state, random
generators, and the scheduled-sampling counter are initialized for that run.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `No module named python` | Change into the repository root before running `python -m ...`; do not run files directly from inside `python/`. |
| PowerShell refuses `Activate.ps1` | Use `.venv\Scripts\python.exe` in place of `python`; activation is optional. |
| CUDA requested but unavailable | Use `--device cpu` for the preview, or install a CUDA build of PyTorch appropriate for your system. |
| CUDA out of memory in training | Start with `--batch-size 1 --val-batch-size 1 --num-workers 0`. Ten-step training retains a large computation graph; even batch 1 may require a larger GPU at the original resolution. |
| Full-resolution CPU prediction is slow | Use the explicit 64-pixel/three-step preview first. It tests installation but has different metrics from the full example. |
| Output directory already contains files | Choose a new `--output-dir`; existing results are deliberately protected. |
| Checkpoint was trained for concentration, not stress | Use `--field concentration` with the supplied weights; train separate stress weights for stress predictions. |
| Split cases do not match the data directory | Provide the same case collection recorded by the checkpoint and reuse its saved split. |
| Fewer frames than required / missing or ambiguous static image | Check the exact folder structure, frame count, and single orientation/C-rate inputs in the [data guide](data-format.md). |
| LiveLink is unavailable or packing fails | Follow the [MATLAB guide](matlab-workflow.md); adding `mli` alone does not connect COMSOL. Inspect `work/.../packing_failure.mat` for a failed packing run. |

The Python regression tests are available with:

```bash
python -m unittest discover -s tests -v
```

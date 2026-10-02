# Running the repository

Start with the [README installation and CPU preview](../README.md#6-run-the-example).
Use Python 3.12 and execute modules from the repository root. You do not need
MATLAB, COMSOL, the full simulation archive, or a GPU for this preview.

## What is included

| Goal | Available in this repository? |
| --- | --- |
| Run the supplied concentration predictor | Yes: pretrained weights and two complete image cases |
| Inspect the 5C prediction without installing anything | Yes: `assets/results/5c/` |
| Exercise both training stages | Yes: the two cases support a small execution check |
| Train a separate stress predictor | Code and example stress images are included; trained stress weights are not |
| Repeat the historical training experiment exactly | No: the complete archive, original split, optimizer state, and training logs are not included |
| Generate a new simulation | Requires separately installed MATLAB, toolboxes, COMSOL, and a connected LiveLink session; see the [MATLAB guide](matlab-workflow.md) |

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
batch-limited smoke run. These briefly trained weights are not intended to
produce the pretrained demonstration's accuracy.

In a fresh CPU environment on the audited i9-13900H machine, the three commands
took approximately 3.5, 4.1, and 3.4 seconds respectively, including Python
startup. These observations exclude installation and do not estimate full
training time. Details and versions are in [reproducibility notes](reproducibility.md).

## Moving to your own data

Use one subdirectory per independent simulation case, following the
[data-format contract](data-format.md). At least two complete cases are needed
for a disjoint train/validation split. Initial training needs at least six
frames per case; the default ten-step sequence stage needs at least fifteen.
More independent cases are necessary for a meaningful generalization study.

Keep concentration and stress runs separate. The supplied checkpoint is verified
as concentration by its SHA-256, including after renaming, and rejects a stress
request. Newly saved checkpoints also record their target field. Unknown legacy
weights without field metadata generate a warning: the loader cannot infer
their training target or original data split.

`--checkpoint` loads weights into a **new** optimization run; it does not restore
Adam moments, random states, or the scheduled-sampling step counter. Use it for
initialization/fine-tuning, not exact recovery of an interrupted training run.

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
| Split cases do not match the data directory | Provide the same case collection recorded by the checkpoint. A new split cannot prove legacy weights have never seen validation cases. |
| Fewer frames than required / missing or ambiguous static image | Check the exact folder structure, frame count, and single orientation/C-rate inputs in the [data guide](data-format.md). |
| LiveLink is unavailable or packing fails | Follow the [MATLAB guide](matlab-workflow.md); adding `mli` alone does not connect COMSOL. Inspect `work/.../packing_failure.mat` for a failed packing run. |

The Python regression tests are available with:

```bash
python -m unittest discover -s tests -v
```

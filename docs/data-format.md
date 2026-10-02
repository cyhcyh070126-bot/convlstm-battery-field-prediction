# Data format and interpretation

For a picture-by-picture explanation of one case, read
[the complete MATLAB–COMSOL sample guide](simulation-sample.md).

## One directory per simulated case

The dataset root contains case directories with the original experiment names, for example:

```text
N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203/
  1_Concentration/concentration_t00000.png ... concentration_t02400.png
  2_Stress/stress_mises_t00000.png ... stress_mises_t02400.png
  3_Voronoi_Geometry/05_Voronoi_Theta_Colored.png
  C-rate/5C.png
```

The folder name records `N`, distribution label, `mu`, `sigma`, `R0`, C-rate, and a case `ID`. The example manifest preserves these encoded values. The ID identifies the simulation case; physical parameters and random seeds are recorded separately by the simulation workflow.

The `05_` geometry image supplies the static orientation input. The examples contain the images used by the Python pipeline; the MATLAB workflow additionally produces geometry illustrations, COMSOL `.mph` files, and `.mat` metadata.

One case's learning-image collection comprises **25 concentration + 25 stress + 1 orientation + 1 C-rate image = 52 PNGs**. A run targeting one field uses
that field's 25 images and the two shared conditioning maps. Four additional
geometry-construction plots bring the complete image export to **56 PNGs**;
case 93203's four plots are in [`assets/sample-5c/`](../assets/sample-5c/).

## Images and time

Each of the two included cases contains 25 concentration frames and 25 von Mises stress frames. All learning inputs are 512 x 512 RGB PNG images. In the source MATLAB export loop, `tlist_values = 0:100:2400`; the five-digit filename suffix therefore records nominal time in seconds. For example, `t00500` is time 500 s, not frame number 500. The image at time 0 is sequence index 0.

Sort the zero-padded filenames chronologically and check that concentration and stress timestamps agree. Keep all windows from a single case together when constructing a train/validation/test split, because neighboring windows share frames and static geometry.

## Nine input channels

Train a separate prediction model for the chosen field. A concentration run uses `1_Concentration`; a stress run uses `2_Stress`. The input for each time step concatenates:

| Channel indices | Content | Changes over time? |
| --- | --- | --- |
| `0:3` | RGB image of the selected field | Yes |
| `3:6` | RGB grain-orientation plot | No |
| `6:9` | RGB C-rate map | No |

Convert the RGB bytes to floating point and divide by 255. A history batch has shape `(batch, history_length, 9, height, width)`. The target has three RGB channels for the selected future field. In autoregressive evaluation, append each predicted field together with the same six static channels before predicting the next step. Spatial patching, when used, must apply identical crop coordinates to the field, orientation, and C-rate images.

The bundled 1C map is uniformly RGB `(0, 0, 255)`; the 5C map is uniformly `(153, 0, 0)`. These agree with the 1C and 5C anchors in the MATLAB C-rate export mapping. The uniform color encodes the charging condition throughout the image.

## Image representation

The model operates on **rendered RGB plots** of concentration, stress, and grain orientation. The field images include grain boundaries and background pixels. The exporter uses separate color tables for concentration and stress, and the model learns those image representations.

The orientation and field images share the same grain topology and retain their original plotting extents: orientation plots nearly fill the image, while field plots include wider outer margins. Orientation therefore provides image-level conditioning in this dataset. For analyses requiring pointwise physical alignment, export all channels with a shared coordinate extent and rendering convention.

## Losses and metrics

The published model uses the image-space objective

```text
loss = MSE(predicted_rgb, reference_rgb)
       + alpha * (1 - SSIM(predicted_rgb, reference_rgb))
```

MSE and SSIM apply to RGB values normalized to `[0, 1]`. Full-image averages include background and boundary pixels and describe image-space agreement. Evaluation in physical units uses scalar field exports together with their coordinates and material masks.

## Scope of the bundled examples

The [two example cases](../examples/README.md) provide complete sequences for file-format checks, training walkthroughs, and checkpoint evaluation. The training walkthrough assigns one case to training and one to validation, keeping all windows from each case together. The README's pretrained 5C evaluation is presented as a case study.

[examples/manifest.json](../examples/manifest.json) lists SHA-256 digests verified against the original source files. It contains repository-relative paths only.

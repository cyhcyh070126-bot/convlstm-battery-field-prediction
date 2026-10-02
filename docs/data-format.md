# Data format and interpretation

## One directory per simulated case

The dataset root contains case directories with the original experiment names, for example:

```text
N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.245_C=1_ID=07415/
  1_Concentration/concentration_t00000.png ... concentration_t02400.png
  2_Stress/stress_mises_t00000.png ... stress_mises_t02400.png
  3_Voronoi_Geometry/05_Voronoi_Theta_Colored.png
  C-rate/1C.png
```

The folder name records `N`, distribution label, `mu`, `sigma`, `R0`, C-rate, and a case `ID`. The example manifest preserves these encoded values. The ID is not an established random seed; physical units for `mu`, `sigma`, and `R0` are not supplied by the folder name.

Only the `05_` geometry image is required as the static orientation input. Other geometry-construction figures, COMSOL `.mph` files, and MATLAB `.mat` files are not included in the small examples.

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

The bundled 1C map is uniformly RGB `(0, 0, 255)`; the 5C map is uniformly `(153, 0, 0)`. These agree with the 1C and 5C anchors in the MATLAB C-rate export mapping. The maps are color encodings of the charging condition, not spatially varying scalar physical fields.

## Rendering limitations

These files are **rendered RGB plots**, not raw concentration, stress, or orientation tensors. The field images include grain boundaries and background pixels. The original exporter uses different color tables for concentration and stress; a model learns those image representations.

Visual inspection of the included examples shows the same grain topology in the orientation and field plots, but the orientation plot nearly fills the image whereas the field plots have wider outer margins. They have not been geometrically registered. The examples preserve the original bytes, so a conditioning pixel is not guaranteed to refer to the same physical location in the field image. Do not interpret the RGB orientation map as a calibrated local diffusion tensor. A workflow requiring physical pointwise alignment should regenerate all channels with a shared coordinate extent and rendering convention.

## Losses and metrics

The published model uses the image-space objective

```text
loss = MSE(predicted_rgb, reference_rgb)
       + alpha * (1 - SSIM(predicted_rgb, reference_rgb))
```

MSE and SSIM apply to RGB values normalized to `[0, 1]`. Unless an evaluation explicitly supplies a material mask, full-image averages include background and boundary pixels. An RGB MSE, RMSE, or SSIM score does not have concentration or stress units and is not a physical relative field error. Quantitative physical validation would require exported scalar fields, coordinate information, masks, and an independently validated mapping.

## Scope of the bundled examples

The [two example cases](../examples/README.md) provide complete sequences for reproducible file-format checks and small training/evaluation smoke runs. Their historical split membership is unknown. Using one case to train and one to validate makes the software exercise possible; it does not establish generalization across microstructures or C-rates. No benchmark claim or pretrained-model accuracy is implied by including these images.

[examples/manifest.json](../examples/manifest.json) lists SHA-256 digests verified against the original source files. It contains repository-relative paths only.

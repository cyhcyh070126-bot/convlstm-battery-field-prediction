# Original-source audit (2026-10-06)

This audit compares each in-scope source file with the author’s supplied local originals. It does not mean every historical version is used by the current quick start.

Exact source bytes are retained in `source_archive/`. Separately reviewed copies are in `research_scripts/`; each change has a [unified diff](source-patches/). The supported portable entry points remain the ones in the repository README.

[File hashes and coverage](complete-source-manifest.json) · [Definition-level review and settings](file-review.json)

## Coverage and current entry points

The original directory contains 48 MATLAB/Python code files. The earlier selected-source manifest covered 13. This audit preserves all 40 files within the requested normal-workflow scope; eight out-of-scope scripts are inventoried but their code is not published. Saved-old, saved-new and training subfolders are kept separate. There are 40 source snapshots, 40 reviewed copies and one explicit compatibility alias.

The selected single-frame original is `训练代码/单帧预测模型/train_pytorch.py`; the selected sequence original is `训练代码/连续预测模型/finetune_sequence.py`. The reviewed root `train_pytorch.py` is an exact alias of the reviewed normal single-frame script, solely to resolve the newer prediction script’s companion import. It is not presented as the original root file.

Use `python.train.single_frame`, `python.train.sequence` and `python.evaluation.predict` for portable training and prediction. Use `matlab/run_workflow.m` for the original selected 2D MATLAB–COMSOL workflow. The old pure-ConvLSTM model and the two distinct FNO tasks are historical alternatives, with different weights and channel contracts.

## Verified comparisons

- 15 Python unit tests passed in a separate CPU environment, including original-versus-packaged model output, MSE + SSIM value, all parameter gradients, real-case patch tensors and rejection of empty exports.
- MATLAB R2025b: exact radius parity for N=40/60/80 and Weibull/Lognormal/Normal (nine combinations, common seed 1234); maximum difference zero in every case.
- Standalone seed handling, five invalid option cases, the original new-version Voronoi renderer, 2D/3D radial corrections and six image-export checks passed.
- Original scientific defaults, rendering constants, normal training objective and selected recurrent formulas are retained. Historical scripts are not silently combined.

## Data collection audit

All 14,659 original PNG paths across 262 cases were inspected using image-header/PNG integrity checks. One incomplete case was found and subsequently removed from the active dataset at the author’s request. The remaining 261 complete cases contain 14,616 verified PNGs: 13,572 are 512 × 512 RGB and 1,044 are 2027 × 1221 RGB construction plots. Each remaining case has 25 concentration and 25 stress frames. This is file-integrity verification, not independent validation of simulation physics.

Removed case: `N=60_Normal_mu=2.00_sigma=0.35_R0=17.416_C=3_ID=24901`. Its `concentration_t01800.png` had zero bytes and subsequent frames were absent. The folder is outside `export_images/`, so training-case discovery no longer reads it; a local isolated copy is retained. No reference image was fabricated. The two bundled example cases are unaffected.

## Verification limits and non-code files

LiveLink helper functions were unavailable in the audit MATLAB session, so a full newly generated COMSOL solve was not executed. The periodic historical plotting helper also depends on `Draw_particle_2D.m` and `Draw_particle_3D.m`, which were not supplied; a search of the author’s Desktop and local GitHub checkouts found no originals. It is not called by the supported circular workflow. The circular dispatcher’s 3D branch remains unsupported; no replacement scientific implementation was invented.

Original FNO scripts require neuraloperator and other original auxiliary dependencies; their full training was not rerun. Historical training scripts also retain their original data/checkpoint paths and hardware assumptions. Configure those before running a historical version, or use the portable README entry points.

The `.asv` backups were compared separately: geometry autosave uses Weibull and sigma=0.1 instead of the saved `.m` Normal/sigma=0.35; simulation autosave uses 4C instead of 5C and comments out model saving. They are autosaves, not replacements for the explicit saved sources. The mojibake backup is not substituted for the valid COMSOL builder. VS caches, bytecode, the local article PDF and local activation text are not source-code deliverables. Existing data/model/weight provenance remains in the dataset and checkpoint records.

## File-by-file review

### Source root

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [Boundary_Particle_back.m](../source_archive/Boundary_Particle_back.m) | Periodic Cartesian box boundary relocation; retained auxiliary, not the selected circular workflow. | None; reviewed copy retains original bytes. |
| [Boundary_Particle_back_circle.m](../source_archive/Boundary_Particle_back_circle.m) | Circular/spherical radial wrapping helper; fix coordinate-update denominator; checked in 2D and 3D. | Compute all radial-correction components from the same pre-update norm; reject impossible inner-boundary radii and allow rounding tolerance. |
| [Boundary_Particle_back_circle_2.m](../source_archive/Boundary_Particle_back_circle_2.m) | Inner circular/spherical projection helper; fix denominator and rounding termination; reject impossible particle radius. | Compute all radial-correction components from the same pre-update norm; reject impossible inner-boundary radii and allow rounding tolerance. |
| [build_full_comsol_model.m](../source_archive/build_full_comsol_model.m) | Selected COMSOL geometry, crystal angles, anisotropic transport/elasticity, swelling, mesh, transient-study setup. | Use the current COMSOL output directory rather than an absent drive. |
| [build_full_comsol_model_recovered_preview.m](../source_archive/build_full_comsol_model_recovered_preview.m) | Alternative recovered COMSOL builder. Restore corrupted display strings from the valid original companion; do not merge alternative model assignments. | Restore corrupted display strings from the valid original companion model builder; model assignments unchanged.; Use the current COMSOL output directory rather than an absent drive. |
| [calculate_force.m](../source_archive/calculate_force.m) | Contact force aggregation and particle force norms; selected packing helper. | None; reviewed copy retains original bytes. |
| [check_fno_structure.py](../source_archive/check_fno_structure.py) | Historical FNO shape check: 21 input channels to 60 output channels; distinct task from train_fno_emoji.py. neuraloperator/torchinfo required; full FNO execution not verified. | Resolve original companion imports from the research-script tree. |
| [check_patching.py](../source_archive/check_patching.py) | Original patch-data inspection; resolve missing dataset name to the existing normal RAMBatteryDataset and calculate expected sample count from actual frames. | Use the existing single-frame patch dataset; derive expected windows from actual frame count; disable augmentation for inspection.; Resolve original companion imports from the research-script tree. |
| [Circle_Boundary.m](../source_archive/Circle_Boundary.m) | Virtual boundary particles for circular packing; used by selected geometry; preserve discretization for valid radii. | Reject radii for which the original boundary discretization divides by zero; valid-radius discretization unchanged. |
| [Contact_Detection_1.m](../source_archive/Contact_Detection_1.m) | Pairwise particle contact detection; historical alternative to the bounding-box detector. | None; reviewed copy retains original bytes. |
| [Contact_Detection_By_Boundary_Box1.m](../source_archive/Contact_Detection_By_Boundary_Box1.m) | Bounding-box filtered particle contacts and overlap information; selected packing detector. | None; reviewed copy retains original bytes. |
| [convlstm.py](../source_archive/convlstm.py) | Actual ConvLSTM recurrent cell, gates and stacked recurrence. Fix advertised scalar/tuple constructor and dtype only; verified original forward/backward equality for the selected model. | Normalize the advertised scalar/tuple constructor options and initialize states in the model dtype; recurrent formulas unchanged.; Resolve original companion imports from the research-script tree. |
| [create_comsol_geometry.m](../source_archive/create_comsol_geometry.m) | Geometry-only COMSOL polygon builder; historical alternative, not the coupled model driver. | None; reviewed copy retains original bytes. |
| [data_loader.py](../source_archive/data_loader.py) | RGB field/static-map loader; concentration, stress and both modes. Fix stress-only frame count and Unicode paths; reject inconsistent frame counts. | Use actual stress frame count for stress-only loading and reject mismatched both-field sequences/unknown modes.; Decode actual image bytes on Unicode paths; preserve original resize, RGB order and normalization.; Resolve original companion imports from the research-script tree. |
| [Draw_particle_1.m](../source_archive/Draw_particle_1.m) | Cartesian particle plotting dispatcher to supplied 2D/3D _1 helpers; auxiliary. | None; reviewed copy retains original bytes. |
| [Draw_particle_2D_1.m](../source_archive/Draw_particle_2D_1.m) | 2D Cartesian particle/box plot; original geometry and plotting constants retained. | None; reviewed copy retains original bytes. |
| [Draw_particle_3D_1.m](../source_archive/Draw_particle_3D_1.m) | 3D Cartesian particle plot; auxiliary, not a 3D COMSOL workflow. | None; reviewed copy retains original bytes. |
| [Draw_particle_Cricle.m](../source_archive/Draw_particle_Cricle.m) | Selected circular dispatcher supports 2D only; original 3D branch is unimplemented and now raises an explicit error. | Report the original unsupported 3D dispatcher explicitly instead of claiming a successful plot; supported 2D drawing unchanged. |
| [Draw_particle_Cricle_2D.m](../source_archive/Draw_particle_Cricle_2D.m) | Selected 2D circular particle plot and original axes limits. | None; reviewed copy retains original bytes. |
| [Draw_particle_Cricle_3D.m](../source_archive/Draw_particle_Cricle_3D.m) | Separate historical 3D circle/sphere renderer with its own signature; not wired into the 2D driver. | None; reviewed copy retains original bytes. |
| [Draw_particle_period.m](../source_archive/Draw_particle_period.m) | Historical periodic plotting requires absent Draw_particle_2D/Draw_particle_3D helpers. Source reviewed; execution remains unavailable without those originals. Not called by the selected workflow. | None; reviewed copy retains original bytes. |
| [Draw_Polycrystal.m](../source_archive/Draw_Polycrystal.m) | Selected clipped Voronoi construction and radius-colored polygon plot; save geometry and original rendering. | None; reviewed copy retains original bytes. |
| [finetune_sequence.py](../source_archive/finetune_sequence.py) | Normal 5-frame input / 10-step scheduled-sampling fine-tuning; byte-identical duplicate of the selected nested sequence script. | Decode actual image bytes on Unicode paths; preserve original resize, RGB order and normalization.; Propagate SSIM errors rather than silently remove/change the objective.; Resolve original companion imports from the research-script tree. |
| [fix_all_crate_colors.m](../source_archive/fix_all_crate_colors.m) | Original batch C-rate RGB correction over export_images; inspect paths before invoking because it rewrites existing conditioning images. | None; reviewed copy retains original bytes. |
| [main_D21origin.m](../source_archive/main_D21origin.m) | Selected radius distributions, contact-force relaxation, virtual boundary, Voronoi geometry and orientation export. N and distribution controls exposed in matlab/run_workflow.m. | Resolve temporary images relative to the actual working directory. |
| [run_single_simulation.m](../source_archive/run_single_simulation.m) | Original COMSOL orchestration, C-rate/time selection and field exports. Portable selected entry is matlab/run_workflow.m; original script preserves its workspace behavior. | Resolve workflow directory from the copied original script, not the author Desktop. |
| [train_fno_emoji.py](../source_archive/train_fno_emoji.py) | Historical FNO concentration-to-stress task: 25 concentration frames plus static maps (81 channels) to 25 stress frames (75 channels), MSE. Separate from ConvLSTM; not a matched checkpoint for the 21-to-60 shape check. | Decode actual image bytes on Unicode paths; preserve original resize, RGB order and normalization.; Reject missing/unreadable images instead of inventing black data.; Reject missing C-rate conditioning rather than synthesize a black input.; Resolve original companion imports from the research-script tree. |

### 保存新代码

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [保存新代码/Draw_Polycrystal.m](../source_archive/保存新代码/Draw_Polycrystal.m) | Saved-new Voronoi plotting version; distinct source retained separately, not substituted into the selected geometry entry. | None; reviewed copy retains original bytes. |
| [保存新代码/predict_pytorch.py](../source_archive/保存新代码/predict_pytorch.py) | Normal newer-model autoregressive prediction; companion import points to the selected Conv3d + ConvLSTM single-frame model, not another root experiment. | Resolve original companion imports from the research-script tree. |

### 保存旧代码

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [保存旧代码/Draw_Polycrystal.m](../source_archive/保存旧代码/Draw_Polycrystal.m) | Saved-old Voronoi plotting version; original plotting choices remain separate. | None; reviewed copy retains original bytes. |
| [保存旧代码/predict_pytorch.py](../source_archive/保存旧代码/predict_pytorch.py) | Old-model autoregressive prediction; local companion import uses 保存旧代码/train_pytorch.py. Old and new model weights are not interchangeable. | Resolve original companion imports from the research-script tree. |
| [保存旧代码/train_pytorch.py](../source_archive/保存旧代码/train_pytorch.py) | Historical pure ConvLSTM single-frame trainer, 50 epochs; different architecture from the selected Conv3d + ConvLSTM model. MSE + SSIM retained. | Resolve original companion imports from the research-script tree. |

### 新版

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [新版/calculate_force.m](../source_archive/新版/calculate_force.m) | Saved-new force helper; independent file hash checked against the selected helper, preserved as its own original. | None; reviewed copy retains original bytes. |
| [新版/Contact_Detection_By_Boundary_Box1.m](../source_archive/新版/Contact_Detection_By_Boundary_Box1.m) | Saved-new bounding-box contact helper; independent file retained with its original implementation. | None; reviewed copy retains original bytes. |
| [新版/Draw_Polycrystal.m](../source_archive/新版/Draw_Polycrystal.m) | Alternative Voronoi renderer; bind the previously undefined ax to the existing axes. Real MATLAB render passed; layout constants unchanged. | Bind ax to the existing figure; preserve every plotting parameter. |

### 训练代码/单帧预测模型

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [训练代码/单帧预测模型/train_pytorch.py](../source_archive/训练代码/单帧预测模型/train_pytorch.py) | Selected normal single-frame training: Conv3d + three ConvLSTM layers, 5-frame history, 128-pixel patches, MSE + SSIM. Model/loss/gradient and real-case patch parity verified. | Decode actual image bytes on Unicode paths; preserve original resize, RGB order and normalization.; Reject missing/unreadable images instead of inventing black data.; Propagate SSIM errors rather than silently remove/change the objective.; Resolve original companion imports from the research-script tree. |

### 训练代码/单帧预测模型/基于权重再训练

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [训练代码/单帧预测模型/基于权重再训练/train_refined_single_frame.py](../source_archive/训练代码/单帧预测模型/基于权重再训练/train_refined_single_frame.py) | Normal single-frame checkpoint refinement with its original learning-rate schedule; not silently merged into the initial training stage. | Decode actual image bytes on Unicode paths; preserve original resize, RGB order and normalization.; Reject missing/unreadable images instead of inventing black data.; Propagate SSIM errors rather than silently remove/change the objective.; Resolve original companion imports from the research-script tree. |

### 训练代码/连续预测模型

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [训练代码/连续预测模型/finetune_sequence.py](../source_archive/训练代码/连续预测模型/finetune_sequence.py) | Selected normal sequence fine-tuning: 5-frame history to 10 future frames, original scheduled sampling and per-step detach behavior retained. | Decode actual image bytes on Unicode paths; preserve original resize, RGB order and normalization.; Propagate SSIM errors rather than silently remove/change the objective.; Resolve original companion imports from the research-script tree. |

### 训练代码/连续预测模型/继续训练

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [训练代码/连续预测模型/继续训练/train_half_start.py](../source_archive/训练代码/连续预测模型/继续训练/train_half_start.py) | Normal continued sequence training with its own epsilon start/decay, learning rate and checkpoint; retained as a separate version. | Decode actual image bytes on Unicode paths; preserve original resize, RGB order and normalization.; Propagate SSIM errors rather than silently remove/change the objective.; Resolve original companion imports from the research-script tree. |

### 训练代码/连续预测模型/继续训练/极致微调

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [训练代码/连续预测模型/继续训练/极致微调/train_phase2_deep_refine.py](../source_archive/训练代码/连续预测模型/继续训练/极致微调/train_phase2_deep_refine.py) | Normal second-phase sequence refinement with its own 100-epoch/StepLR configuration; retained separately. | Decode actual image bytes on Unicode paths; preserve original resize, RGB order and normalization.; Propagate SSIM errors rather than silently remove/change the objective.; Resolve original companion imports from the research-script tree. |

## Re-run the original MATLAB checks

From the repository root in MATLAB R2025b with the listed geometry toolboxes:

```matlab
addpath(fullfile(pwd, 'tests', 'matlab'));
audit_original_sources;
```

This compares radius samples from the preserved originals, exercises the reviewed helpers and runs the export checks without COMSOL. It writes a detailed report under `outputs/source-audit-matlab/`. The [recorded verification](audit-verification.json) separates preserved-source analyzer errors from reviewed copies and supported MATLAB files.

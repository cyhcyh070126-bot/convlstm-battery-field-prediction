# Third-party notices

## ConvLSTM implementation

The cell and stacked recurrent-network structure in `python/models/convlstm.py`
is adapted from [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch),
copyright (c) 2017 Andrea Palazzi, under the MIT License. The full upstream
notice is retained in `third_party/ConvLSTM_pytorch-LICENSE`.

The original research copy had the same gate equations, parameter naming,
layer organization, and hidden-state initialization. Packaging retains these
structures so that compatible state dictionaries can be loaded.

## Dependencies

PyTorch, NumPy, OpenCV, pytorch-msssim, Matplotlib, and Pillow are installed as
dependencies, rather than bundled as source. Their respective licenses apply.
MATLAB and COMSOL are separately installed commercial software and are not
distributed with this repository.

## Existing homepage illustrations

The README reuses these image files already present on the author's homepage:

- `battery_convlstm_cell_clean.png`: corresponding to Figure 4;
- `battery_convlstm_pipeline.png`: corresponding to Figure 5;
- `battery_scheduled_sampling.png`: corresponding to Figure 6.

Their source is Zehou Wang, Ying Zhao, Zheng Zhong, and Bai-Xiang Xu,
*Deep-learning based prediction of chemo-mechanics and damage in battery
active materials*, Energy Storage Materials 82 (2025), 104581,
[doi:10.1016/j.ensm.2025.104581](https://doi.org/10.1016/j.ensm.2025.104581).
The supplied article identifies copyright 2025 Elsevier B.V., all rights
reserved. These images retain their source attribution and are not licensed
under the MIT notice for the ConvLSTM code. No separate reuse license is
granted by this repository.

They are conceptual illustrations. The scheduled-sampling image contains
token/softmax notation; the implementation in this repository instead performs
continuous RGB regression. The actual architecture is documented in the README.

## Project-specific material

This initial release does not assign a blanket open-source license to the
project-specific research code or dataset. The upstream MIT notice above
applies to the identified ConvLSTM component. Contact the repository owner
through GitHub for permissions concerning other material.

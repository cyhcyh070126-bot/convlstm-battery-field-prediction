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

## Project-specific material

This initial release does not assign a blanket open-source license to the
project-specific research code or dataset. The upstream MIT notice above
applies to the identified ConvLSTM component. Contact the repository owner
through GitHub for permissions concerning other material.

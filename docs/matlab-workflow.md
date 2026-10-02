# MATLAB and COMSOL data generation

The MATLAB component generates a circular, two-dimensional polycrystalline
microstructure, builds a coupled COMSOL model, solves the transient problem, and
exports the rendered concentration and von Mises stress sequences used by the
Python image-prediction pipeline.

## Files and execution order

```text
matlab/run_workflow.m
  -> simulation/run_single_simulation.m
     -> geometry/main_D21origin.m
        -> contact, force, boundary, and plotting helpers in geometry/
     -> simulation/build_full_comsol_model.m
     -> export/export_simulation_images.m
```

The geometry and model-construction scripts retain the original algorithms,
parameter values, and source comments. The original single simulation driver
has been separated into orchestration and export. Portability changes remove
machine-specific paths and workspace-clearing commands, isolate intermediate
files, expose C-rate and the random seed, and propagate model-building errors.
No COMSOL model or training loss has been replaced by a new formulation.

## Requirements

- MATLAB. The source uses implicit expansion, `string`, and `exportgraphics`;
  an exact minimum supported release has not been established.
- Statistics and Machine Learning Toolbox for `normrnd`, `lognrnd`, `wblrnd`,
  and `randsample` in the radius-distribution generator.
- Image Processing Toolbox for `imresize` of the orientation image.
- A licensed COMSOL Multiphysics installation with **LiveLink for MATLAB** and
  a connected COMSOL session. The model uses **Transport of Diluted Species**,
  **Solid Mechanics**, anisotropic linear elasticity, and **Hygroscopic
  Swelling**. The installed license must enable these interfaces and features;
  the source does not establish one definitive module bundle.

The code calls COMSOL's Java model API and `mphsave`. Adding an `mli` directory
to the MATLAB path makes the MATLAB helper functions available; it does not
start or connect a COMSOL server by itself. Use the installation's **COMSOL
with MATLAB** launcher, or establish a LiveLink connection before running.
Keep a separate COMSOL session for this workflow: the inherited model builder
removes any existing model with the tag `Model` in its connected session.

## Run one simulation

From a MATLAB session connected to COMSOL, with the repository as the current
directory:

```matlab
addpath('matlab');
run_dir = run_workflow(fullfile(pwd, 'outputs', 'matlab'));
```

The default C-rate is 5, and the geometry uses `rng('shuffle')`, matching the
original driver. An explicit seed makes the MATLAB random geometry repeatable
within a compatible numerical environment:

```matlab
addpath('matlab');
output_dir = fullfile(pwd, 'outputs', 'matlab');
run_dir = run_workflow(output_dir, '', 5, 42);
```

If LiveLink is connected but its MATLAB helpers are not yet on the path:

```matlab
% Replace comsol_root with your own COMSOL installation directory.
comsol_root = 'C:\Program Files\COMSOL\COMSOL64';
mli_dir = fullfile(comsol_root, 'Multiphysics', 'mli');
run_dir = run_workflow(output_dir, mli_dir, 5, 42);
```

Paths added by the wrapper and the original working directory are restored
when it finishes or errors. Intermediate geometry and the unsolved model are
kept under `outputs/matlab/work/` for inspection; the output directory is not
automatically deleted.

## Original settings

| Setting | Value in the selected source |
| --- | --- |
| Spatial model | 2D geometry with COMSOL length unit `um` |
| Initial particle count | `N = 60` |
| Radius distribution | Normal (`command = 3`), mean 2, standard deviation 0.35 |
| Initial packing fraction | `phi_0 = 0.95` |
| Packing iteration cap | `Tk = 200000` |
| Orientation | `theta ~ Uniform(0, pi/2)`; `beta = atan2(y,x) + theta` |
| Study times | `0:100:2400` seconds: 25 frames including time zero |
| Default simulation C-rate | 5, overriding the builder's initial value |
| COMSOL mesh setting | `autoMeshSize(2)` |
| Concentration rendering | Rainbow color table, upper color limit `4.5E4` |
| Stress rendering | `solid.mises`, Prism color table, upper color limit `5E8` |
| Field export size | 512 × 512 pixels |

Edit the clearly named parameters near the start of
`matlab/geometry/main_D21origin.m` to generate other particle ensembles. Its
Weibull and lognormal branches are also retained. The export time vector and
COMSOL study list live together in `run_single_simulation.m`; keep them
consistent if changing the temporal resolution.

The rendered orientation image has the historical filename
`05_Voronoi_Theta_Colored.png`, but the selected source colors it by the folded
**beta** crystal orientation. `04_Voronoi_Uncolored.png` is also a historical
name: the helper colors grains by particle radius and includes a colorbar.

## Output and Python hand-off

Each call creates one sequence directory:

```text
outputs/matlab/
├── work/<unique-work-directory>/
│   ├── geometry_for_comsol.mat
│   ├── comsol_MERGED_SCIENTIFIC_model.mph
│   └── temp_figs/
└── export_images/
    └── N=60_Normal_mu=2.00_sigma=0.35_R0=..._C=5_ID=.../
        ├── 1_Concentration/
        │   ├── concentration_t00000.png
        │   ├── concentration_t00100.png
        │   └── ... concentration_t02400.png
        ├── 2_Stress/
        │   ├── stress_mises_t00000.png
        │   ├── stress_mises_t00100.png
        │   └── ... stress_mises_t02400.png
        ├── 3_Voronoi_Geometry/
        │   ├── 01_Radius_Distribution.png
        │   ├── 02_Initial_Placement.png
        │   ├── 03_Final_Placement.png
        │   ├── 04_Voronoi_Uncolored.png
        │   └── 05_Voronoi_Theta_Colored.png
        ├── C-rate/5C.png
        ├── geometry_for_comsol.mat
        ├── run_metadata.mat
        └── <run-folder-name>.mph
```

Point the Python data-root option at `outputs/matlab/export_images`, which
contains the individual run directories. The concentration and stress files
are **rendered RGB images**, not raw nodal concentration or stress arrays.
Python image normalization must not be described as normalization of physical
field values. The C-rate input is a 512 × 512 solid RGB image: its color is
linearly interpolated between anchors at 0.5, 1, 2, 3, 4, and 5C, with values
outside that interval clamped to the endpoint colors.

`geometry_for_comsol.mat` stores `V`, `C_finite`, `polygons`, `Ps_finite`, and
`R0`. In the 2D workflow, the columns of `Ps_finite` are x, y, radius, beta, and
theta. `run_metadata.mat` records the requested seed, C-rate, time vector,
particle count, final radius, distribution name, mean, and standard deviation.
Large simulation outputs and COMSOL `.mph` files are generated locally,
not required to browse the repository or run Python on the included example.

## Reproducibility limits

The source is a research workflow, not a validated general-purpose COMSOL
package. Its existing sequential assignment of grain domains after the Boolean
geometry operations and its original boundary selections are preserved. Check
domain numbering, selections, units, and mesh convergence in your COMSOL
version before using newly generated data for scientific conclusions.

MATLAB Code Analyzer inspected all 11 `.m` files with MATLAB R2025b
(25.2.0.2998904); it found no syntax errors. Small analytical checks passed for
contact-pair detection, repulsive force direction and magnitude, noncontact
force handling, and circular boundary geometry. Statistics and Machine
Learning Toolbox and Image Processing Toolbox 25.2 were available during
these checks. This is limited validation of code structure and selected
helpers, not a full solver reproducibility result.

The stochastic packing stage can require up to 200,000 iterations per
microstructure. A complete transient solve was not rerun during repository
preparation, and COMSOL API/license compatibility is not claimed from static
inspection alone. Repository validation records are described in the main
README.

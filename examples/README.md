# Example simulation images

This directory contains two complete, original image sequences from the MATLAB/COMSOL workflow. The PNG files are copied without editing or resampling. They let you inspect the input format and exercise data loading, training, and rollout code on a small dataset.

| Case ID | C-rate encoded in folder name | Concentration frames | Stress frames | Total PNG bytes |
| --- | --- | --- | --- | --- |
| `07415` | 1C | 25 | 25 | 7,854,202 |
| `93203` | 5C | 25 | 25 | 10,928,834 |

Each case also includes one grain-orientation image and one C-rate image. All 104 PNGs are RGB, 512 x 512 pixels. The combined image payload is 18,783,036 bytes (about 17.9 MiB).

## Intended use

Use `examples/data` as the dataset root. The two cases permit a minimal split with one case for training and the other for validation. This is a **toy smoke run**, not a meaningful estimate of predictive performance. Historical training, validation, or test membership is unknown; neither case is identified as a held-out benchmark. Do not split overlapping windows from the same case between training and validation.

The bundled files contain simulation references and static conditioning images only. They do not contain model predictions or pretrained weights. See the [main README](../README.md) for the supported training and evaluation commands.

## Directory layout

```text
data/
  N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.245_C=1_ID=07415/
    1_Concentration/
      concentration_t00000.png
      ...
      concentration_t02400.png
    2_Stress/
      stress_mises_t00000.png
      ...
      stress_mises_t02400.png
    3_Voronoi_Geometry/
      05_Voronoi_Theta_Colored.png
    C-rate/
      1C.png
  N=60_Lognormal_mu=2.00_sigma=0.10_R0=17.288_C=5_ID=93203/
    ...
```

The filename suffix records nominal simulation time in seconds, from 0 to 2400 in increments of 100, as used by the MATLAB exporter. Read the [data format notes](../docs/data-format.md) before interpreting colors or metrics: these are rendered RGB images, and the orientation and field plots use different margins.

## Integrity and provenance

[manifest.json](manifest.json) records every included image's relative path, byte count, dimensions, and SHA-256 digest. Paths are relative to this `examples` directory. Each destination digest was checked against the source image during copying. Case parameters are transcribed from directory names; units or RNG seeds are not inferred from those names.

To check the published image files from the repository root:

```python
import hashlib
import json
from pathlib import Path

root = Path("examples")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for case in manifest["cases"]:
    for item in case["files"]:
        path = root / item["path"]
        assert path.stat().st_size == item["bytes"], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
print(f"Verified {manifest['file_count']} original PNG files.")
```

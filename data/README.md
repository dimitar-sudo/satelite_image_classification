# Dataset

This project uses the **EuroSAT** dataset — 27,000 georeferenced 64×64 RGB image
patches extracted from Sentinel-2 satellite imagery, labelled across 10 land-use
and land-cover classes. The raw images are **not** version-controlled (they are
large); this folder only documents the expected layout.

## Expected layout

```
data/
├── train/                     # one sub-folder per class (labelled)
│   ├── AnnualCrop/
│   ├── Forest/
│   ├── HerbaceousVegetation/
│   ├── Highway/
│   ├── Industrial/
│   ├── Pasture/
│   ├── PermanentCrop/
│   ├── Residential/
│   ├── River/
│   └── SeaLake/
└── test/                      # flat folder of unlabelled images for inference
    ├── 2.jpg
    ├── 101.jpg
    └── ...
```

The training set used here contains 1,000 images per class (10,000 total) and a
held-out flat test set of 1,980 unlabelled images. The class names and the
80/20 stratified train/validation split are defined in
[`configs/default.yaml`](../configs/default.yaml).

## Obtaining the data

The EuroSAT dataset is publicly available:

- Project page: https://github.com/phelber/EuroSAT
- Reference: Helber et al., *"EuroSAT: A Novel Dataset and Deep Learning Benchmark
  for Land Use and Land Cover Classification"*, IEEE JSTARS, 2019.

Download the RGB version, then arrange the files to match the layout above. The
class sub-folder names must match those in the config exactly (they determine
the numeric label ordering).

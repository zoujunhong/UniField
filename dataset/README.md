# Dataset Organization

`dataset/` is organized around query-style field prediction.

## Directory Rules

- `utils/` contains reusable, dataset-agnostic helpers such as VTK loading, geometry extraction, sampling, and normalization.
- `surface/` contains surface-side dataset components. These components read surface cache files and return encoder inputs, surface query coordinates, and surface targets.
- `volume/` contains volume-side dataset components. These components read volume cache files and return volume query coordinates and volume targets.
- `cache/` contains cache builders for raw CFD files. Training code should read cache files instead of raw VTK files.
- Root-level dataset files compose one surface component and one volume component into a full PyTorch `Dataset`.
- The only maintained root dataset files are `DrivAerNet.py`, `CRH450.py`, and `Maglev.py`. Each file must export a `Dataset` alias so `dataset.build_dataset(...)` can load it by name.

## Unified Return Interface

Every composed dataset returns exactly seven values:

```python
surface_input, surface_query, surface_target, volume_query, volume_target, cond, route
```

- `surface_input`: encoder input point cloud. The first three channels are always `xyz`; optional channels are configured by `surface_input_list`, for example `["xyz", "normal", "area"]`.
- `surface_query`: surface query coordinates with shape `(Nsurface, 3)`.
- `surface_target`: target fields for surface queries, configured by `surface_target_list`.
- `volume_query`: volume query coordinates with shape `(Nvolume, 3)`.
- `volume_target`: target fields for volume queries, configured by `volume_target_list`.
- `cond`: free-stream or dataset condition vector.
- `route`: integer route id for multi-domain models.

If surface or volume data is disabled, the corresponding query and target tensors are zero-length tensors rather than `None`.

## Field Lists

Fields are selected by list arguments:

```python
surface_input_list=["xyz", "normal", "area"]
surface_target_list=["p"]
volume_target_list=["U"]
```

Scalar fields add one channel. `U`, `xyz`, and `normal` add three channels.

## Adding A Dataset

1. Add shared parsing/cache helpers under `utils/` or `cache/` only if they are reusable.
2. Add a surface component under `surface/` and a volume component under `volume/`, even if one side only returns empty tensors.
3. Add a root-level composed dataset file that owns the PyTorch `Dataset` interface and returns the unified seven-value tuple.
4. Export `Dataset = YourDatasetClass` in the root dataset file. Dynamic loading will find it by file name.

## Config-Driven Construction

Datasets are built from Python config dictionaries:

```python
CONFIG = {
    "name": "DrivAerNet",
    "kwargs": {...},
}
```

Multiple datasets can be merged:

```python
CONFIG = {
    "name": ["CRH450", "Maglev"],
    "kwargs": {
        "CRH450": {...},
        "Maglev": {...},
    },
}
```

Merged datasets must have identical `surface_input_list`, `surface_target_list`, `volume_target_list`, and sample counts.

Dataset option configs should describe the data shape, fields, sampling counts, normalization, and route id. Stage-specific splits belong in the experiment config:

```python
"train": {"dataset_kwargs": {"ids_file": ".../train_design_ids.txt"}},
"test": {"dataset_kwargs": {"ids_file": ".../val_design_ids.txt"}},
```

For CRH450 and Maglev, `{"split": "train"}` selects cases by geometry id: CRH450 `geo_1`-`geo_18` and Maglev `geo_1`-`geo_27`. `{"split": "test"}` or `{"split": "val"}` selects CRH450 `geo_19`-`geo_20` and Maglev `geo_28`-`geo_30`. Incomplete or partially written high-speed cache files are skipped by default.

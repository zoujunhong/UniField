# UniField

[Paper information and citations](PaperInformation.md)

This repository contains query-style neural field models for aerodynamic field prediction. It now hosts three related model lines:

- **AdaField**: surface pressure modeling with SAPT and flow-conditioned adaptation.
- **UniFieldV1**: first multi-domain UniField surface pressure model with parallel route adapters.
- **UniFieldV2**: Swin-like one-dimensional window attention UniField used by the DrivAerNet demo checkpoint.
- **QueryUniField**: the current encoder-decoder query framework for surface and volume field prediction.

The codebase is organized around config-driven training. Datasets, models, losses, checkpointing, and visualization utilities are separated so experiments can be reproduced by changing a single Python config file.

## Repository Layout

```text
config/          # dataset/model/experiment Python configs
dataset/         # composed datasets plus surface/volume/cache helpers
model/           # encoders, decoders, query model parent, reusable model utils
utils/           # training, checkpoint, scheduler, task, visualization helpers
loss/            # optional physics/operator losses when present
train.py         # config-driven training entrypoint
test.py          # config-driven evaluation entrypoint
visualization.py # config-driven visualization entrypoint
```

Generated outputs such as `demo/`, `log/`, `temp/`, checkpoints, caches, and local Codex metadata are ignored by Git.

## Quick Start

All main entrypoints accept only one argument:

```bash
python train.py --config config/experiment/DrivAerNet_AdaField_surfaceP.py
python test.py --config config/experiment/DrivAerNet_AdaField_surfaceP.py
python visualization.py --config config/experiment/DrivAerNet_AdaField_surfaceP.py
```

Current example experiments include:

- `config/experiment/DrivAerNet_AdaField_surfaceP.py`
- `config/experiment/DrivAerNet_UniFieldV1_surfaceP.py`
- `config/experiment/DrivAerNet_UniFieldV2_surfaceP.py`
- `config/experiment/DrivAerNet_UniFieldCrossAttention_surfaceP_volumeU.py`

## Model System

Models are assembled from one encoder and one decoder:

```python
CONFIG = {
    "encoder_name": "AdaFieldEncoder",
    "encoder_kwargs": {...},
    "decoder_name": "SurfacePressureDecoder",
    "decoder_kwargs": {...},
}
```

`model.build_model(config)` resolves the encoder from `model/Encoder`, the decoder from `model/Decoder`, and composes them with `QueryUniField`.

Maintained encoder families:

- `AdaFieldEncoder`: legacy AdaField SAPT backbone converted to dense surface memory.
- `UniFieldV1Encoder`: legacy multi-route UniField backbone converted to dense surface memory.
- `UniFieldV2Encoder`: legacy Swin-like window-attention UniField encoder converted to query memory.
- `UniFieldEncoder`: current flow/manifold-aware encoder.

Maintained decoder families:

- `SurfacePressureDecoder`: surface-only pressure query decoder for AdaField and UniFieldV1.
- `UniFieldV2Decoder`: legacy UniFieldV2 U-Net decoder adapted for surface query prediction.
- `CrossAttentionDecoder`, `AnchorDecoder`, `MLPQueryDecoder`, `SelfAttentionDecoder`: query decoders for the current QueryUniField line.

## Dataset System

Datasets return a stable seven-value tuple:

```python
surface_input, surface_query, surface_target, volume_query, volume_target, cond, route
```

`surface_input` is used by the encoder. `surface_query` and `volume_query` are used for supervised query prediction. Disabled domains return zero-length tensors rather than `None`.

Maintained root dataset names:

- `DrivAerNet`
- `CRH450`
- `Maglev`

The old FlowBench path from the first UniField repository is not migrated into this mainline.

## Configs

Config files are Python files exporting `CONFIG`.

- Dataset configs live under `config/dataset/<Dataset>/<option>.py`.
- Model configs live under `config/model/<Model>/<Size>.py`.
- Experiment configs live under `config/experiment/*.py`.

See [config/README.md](config/README.md) for detailed fields, normalization conventions, and denormalization options.

## Checkpoints

Legacy AdaField and UniFieldV1 checkpoints are loaded through the current checkpoint utilities. The loader adds compatibility aliases:

- old backbone keys such as `proj_in.*`, `tf_down_layers.*`, and `adapt_*` map to `encoder.*`;
- old `proj_out.*` maps to `decoder.proj_out.*`;
- UniFieldV1 route branches remain compatible with `route_map`.

Use `checkpoint` and `route_map` fields in experiment configs for pretraining or transfer.

## Notes

- AdaField and UniFieldV1 are kept as surface-pressure models.
- The latest QueryUniField line supports richer query tasks such as surface pressure plus volume velocity.
- Raw VTK files should be converted to cache before training; training code should not read raw VTK files in the hot path.

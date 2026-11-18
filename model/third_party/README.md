# Third-party model implementations

This directory contains the upstream implementations used by the retained
models and historical main experiments.

- `utonia_dsapt/` contains the Utonia backbone used by DSAPT, with its
  Apache-2.0 license and the experiment's parameter layout preserved.
- `abupt/`, `smart/`, and `transolver3/` contain the official baseline
  implementations; ABUPT and SMART retain their source license files. The adapters in
  `model/Baseline/` provide the common field-model interface.

Pretrained weights live outside the repository under
`~/zoujunhong/model/pretrain/`. Historical downloaded weights are retained
there even when their model family is no longer registered. See
`CHECKPOINTS.md` for original download provenance and `model/README.md` for
the supported model interfaces.

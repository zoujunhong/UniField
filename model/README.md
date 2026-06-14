# Model Organization

`model/` uses a composable query-model layout:

```text
model/
  utils/      # reusable small layers, attention, downsampling, checkpoint helpers
  Encoder/    # surface/geometry encoders
  Decoder/    # query decoders that interact with encoder memory
  query_unifield.py
```

## Rules

- Reusable network pieces live in `model/utils/`. Do not define local copies of FFN, AdaLN, attention, or downsampling blocks inside concrete models.
- `Encoder/` modules convert surface representations into memory dictionaries. Current memory keys are `features`, `points`, `normals`, `area`, `log_area`, and `flow`.
- `Decoder/` modules consume encoder memory and query points. They should expose:
  - `forward(query_pos, memory, cond, routes, flow)`
  - `prepare_memory(memory, cond, routes, flow)`
  - `decode_with_memory(query_pos, prepared, cond, routes, flow)` when chunked inference is supported.
- `query_unifield.py` is the only complete-model parent class. It receives one encoder and one decoder, then handles `forward`, `forward_test`, and route-compatible weight loading.
- `model.build_model(config)` builds the model from `encoder_name`, `encoder_kwargs`, `decoder_name`, and `decoder_kwargs`.
- `decoder_kwargs["memory_dim"]` can be omitted; the builder infers it from `encoder_kwargs["embed_dims"][-1]`.

## Naming

- `AdaLN`: one conditional LayerNorm branch.
- `RoutedAdaLN`: a route-aware wrapper that owns multiple `AdaLN` branches.
- `FFN`: the plain transformer feed-forward network.
- `LinearNorm`: linear projection followed by normalization and optional activation.
- Legacy surface encoders are named by paper/model line: `AdaFieldEncoder` and `UniFieldV1Encoder`.
- Decoder names describe interaction: `AnchorDecoder`, `MLPQueryDecoder`, `CrossAttentionDecoder`, `SelfAttentionDecoder`, `SurfacePressureDecoder`.

## Model Config

Complete models are assembled by config rather than separate root wrapper files:

```python
CONFIG = {
    "encoder_name": "UniFieldEncoder",
    "encoder_kwargs": {...},
    "decoder_name": "CrossAttentionDecoder",
    "decoder_kwargs": {...},
    "default_query_chunk_size": 50000,
}
```

The class named by `encoder_name` must be exported by `model/Encoder/__init__.py`. The class named by `decoder_name` must be exported by `model/Decoder/__init__.py`.

The existing config families keep names such as `UniField_cross_attention` and `UniField_anchor` only as convenient experiment presets; they no longer correspond to root model Python files.

Legacy model presets such as `AdaField` and `UniFieldV1` follow the same config format. They pair `AdaFieldEncoder` or `UniFieldV1Encoder` with `SurfacePressureDecoder`, preserving the old surface-pressure semantics while fitting the current QueryUniField interface.

`SelfAttentionDecoder` does not support chunk-equivalent `forward_test`, because query tokens interact with each other.

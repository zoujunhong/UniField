CONFIG = {
    "encoder_name": "AdaFieldEncoder",
    "encoder_kwargs": {
        "depth": (4, 4, 6, 12, 8),
        "channels": (128, 256, 612, 1024, 2048),
        "num_points": (1024, 256, 64, 16),
        "adapter_dim": 64,
        "cond_dim": 2,
        "k": 16,
    },
    "decoder_name": "SurfacePressureDecoder",
    "decoder_kwargs": {
        "out_channels": 1,
        "interp_k": 3,
        "query_in_dim": 3,
    },
    "default_query_chunk_size": 50000,
}

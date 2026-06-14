CONFIG = {
    "encoder_name": "UniFieldV1Encoder",
    "encoder_kwargs": {
        "depth": (4, 4, 6, 12, 8),
        "channels": (64, 128, 256, 512, 1024),
        "num_points": (1024, 256, 64, 16),
        "adapter_dim": 64,
        "cond_dim": 2,
        "num_routes": 3,
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

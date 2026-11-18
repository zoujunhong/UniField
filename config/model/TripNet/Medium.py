CONFIG = {
    "encoder_name": "TripNetTriplaneEncoder",
    "encoder_kwargs": {
        "triplane_channels": 32,
        "resolution": 128,
        "feature_channels": 32,
        "base_channels": 96,
        "channel_multipliers": (1, 2, 4),
        "cond_dim": 2,
        "num_routes": 1,
        "memory_dim": 96,
    },
    "decoder_name": "TripNetQueryDecoder",
    "decoder_kwargs": {
        "out_channels": 7,
        "cond_dim": 2,
        "hidden_dim": 256,
        "depth": 4,
        "dropout": 0.0,
        "include_query_coords": True,
        "padding_mode": "zeros",
    },
    "default_query_chunk_size": 50000,
}

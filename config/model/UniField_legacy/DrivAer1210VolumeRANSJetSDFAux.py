CONFIG = {
    "encoder_name": "LegacyUniField1210Encoder",
    "encoder_kwargs": {
        "in_channels": 3,
        "embed_dims": (256, 512, 1024, 2048),
        "depths": (4, 4, 4, 4),
        "num_heads": (2, 4, 8, 8),
        "patch_sizes": (1024, 512, 256, 128),
        "down_ratios": (4, 4, 4),
        "cond_dim": 2,
        "num_routes": 1,
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.0,
    },
    "decoder_name": "CrossAttentionDecoder",
    "decoder_kwargs": {
        # Channel layout:
        # 0:7    U, p, k, omega, nut
        # 7:36   JetNS derivative channels:
        #        gradU8, Hessian(U)18, grad(p)3
        # 36:37  auxiliary surface_distance / wall_distance prediction
        "out_channels": 37,
        "query_dim": 1024,
        "cond_dim": 2,
        "num_routes": 1,
        "query_in_dim": 3,
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.0,
        "proj_drop": 0.0,
        "attn_drop": 0.0,
        "use_flow_ffn": False,
        "depth": 4,
        "num_heads": 8,
        "use_flow_attention": False,
    },
    "default_query_chunk_size": 50000,
}

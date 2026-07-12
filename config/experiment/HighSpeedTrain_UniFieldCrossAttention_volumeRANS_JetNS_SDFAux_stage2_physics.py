from copy import deepcopy

from config.experiment.HighSpeedTrain_UniFieldCrossAttention_volumeRANS_JetNS_SDFAux_stage1_data import (
    CONFIG as STAGE1_CONFIG,
)


CONFIG = deepcopy(STAGE1_CONFIG)

CONFIG["runtime"]["distributed"]["port"] = 45338

CONFIG["task"]["losses"] = [
    {
        "domain": "volume",
        "field": "U",
        "pred_slice": (0, 3),
        "weight": 1.0,
        "mae_weight": 1.0,
        "mse_weight": 1.0,
    },
    {
        "domain": "volume",
        "field": "p",
        "pred_slice": (3, 4),
        "weight": 1.0,
        "mae_weight": 1.0,
        "mse_weight": 1.0,
    },
    {
        "domain": "volume",
        "field": "k",
        "pred_slice": (4, 5),
        "weight": 1.0,
        "mae_weight": 1.0,
        "mse_weight": 1.0,
    },
    {
        "domain": "volume",
        "field": "omega",
        "pred_slice": (5, 6),
        "weight": 1.0,
        "mae_weight": 1.0,
        "mse_weight": 1.0,
    },
    {
        "domain": "volume",
        "field": "nut",
        "pred_slice": (6, 7),
        "weight": 1.0,
        "mae_weight": 1.0,
        "mse_weight": 1.0,
    },
    {
        "domain": "volume",
        "field": "surface_distance",
        "pred_slice": (36, 37),
        "weight": 1.0,
        "mae_weight": 1.0,
        "mse_weight": 1.0,
    },
    {
        "domain": "volume_relation",
        "field": "k_omega_nut_sst_branch",
        "weight": 0.2,
        "equality_tol": 1e-3,
        "mask_denom": "target_nut",
        "omega_eps": 1e-4,
        "loss_type": "l1",
        "hinge_loss_type": "l1",
        "equality_weight": 1.0,
        "upper_bound_weight": 0.25,
        "positive_weight": 0.01,
    },
    {
        "domain": "volume_relation",
        "field": "rans_nonnegative",
        "weight": 0.01,
        "fields": ("k", "omega", "nut"),
    },
    {
        "domain": "volume_relation",
        "field": "jet_taylor_up",
        "weight": 1.0,
        "length_scale": 1.0,
        "pair_sample_size": 1024,
        "pair_candidate_size": 8192,
        "min_pair_distance": 1e-8,
        "symmetric": True,
        "detach_field_values": True,
        "max_residual": 20.0,
        "u_weight": 1.0,
        "p_weight": 1.0,
        "loss_type": "smooth_l1",
        "beta": 0.02,
    },
    {
        "domain": "volume_relation",
        "field": "local_constant_nu_momentum_ratio",
        "weight": 0.1,
        "length_scale": 1.0,
        "nu_lam": 1.5e-5,
        "clamp_nut": True,
        "pressure_grad_eps_factor": 1e-3,
        "loss_type": "l1",
        "max_ratio": 20.0,
    },
]

CONFIG["train"].update(
    {
        "lr": 4e-5,
        "total_epoch": 200,
        "saveroot": (
            "/data/home/zdhs0017/zoujunhong/model/savemodel/field/"
            "HighSpeedTrain_UniField_volumeRANS_JetNS_SDFAux_stage2_physics"
        ),
        "checkpoint": (
            "/data/home/zdhs0017/zoujunhong/model/savemodel/field/"
            "HighSpeedTrain_UniField_volumeRANS_JetNS_SDFAux_stage1_data/"
            "model_epoch_500.pth"
        ),
        "partial_load": True,
        "route_map": None,
        "dry_run_iters": 0,
    }
)

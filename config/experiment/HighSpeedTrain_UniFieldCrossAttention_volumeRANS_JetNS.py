from __future__ import annotations

from copy import deepcopy

from config.experiment.HighSpeedTrain_UniFieldCrossAttention_volumeRANS import (
    CONFIG as BASE_CONFIG,
)


CONFIG = deepcopy(BASE_CONFIG)

CONFIG["model_size"] = "DrivAer1210VolumeRANSJet"

for item in CONFIG["task"]["losses"]:
    if item.get("domain") == "volume" and item.get("field") in {"U", "p", "k", "omega", "nut"}:
        item["mae_weight"] = 1.0
        item["mse_weight"] = 1.0

CONFIG["task"]["losses"].extend(
    [
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
            "weight": 0.05,
            "length_scale": 1.0,
            "pair_sample_size": 1024,
            "pair_candidate_size": 8192,
            "min_pair_distance": 1e-8,
            "symmetric": True,
            "u_weight": 1.0,
            "p_weight": 1.0,
            "loss_type": "smooth_l1",
            "beta": 0.02,
        },
        {
            "domain": "volume_relation",
            "field": "local_constant_nu_momentum_ratio",
            "weight": 0.01,
            "length_scale": 1.0,
            "nu_lam": 1.5e-5,
            "clamp_nut": True,
            "pressure_grad_eps_factor": 1e-3,
            "loss_type": "smooth_l1",
            "beta": 0.1,
            "max_ratio": 20.0,
        },
    ]
)

CONFIG["train"]["saveroot"] = (
    "/data/home/zdhs0017/zoujunhong/model/savemodel/field/"
    "HighSpeedTrain_UniFieldLegacy1210_volumeRANS_JetNS"
)
CONFIG["train"]["lr"] = 1e-5
CONFIG["train"]["total_epoch"] = 100
CONFIG["train"]["resume_epoch"] = 0
CONFIG["train"]["checkpoint"] = (
    "/data/home/zdhs0017/zoujunhong/model/savemodel/field/"
    "HighSpeedTrain_UniFieldLegacy1210_volumeRANS_MAERMSE/model_epoch_400.pth"
)
CONFIG["train"]["partial_load"] = True
CONFIG["runtime"]["distributed"]["port"] = 45335

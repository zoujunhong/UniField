"""Reproducible high-speed train TripNet volume-field experiment.

This config trains TripNet on the combined CRH450 + Maglev high-speed dataset.
It expects the per-case FP32 triplane cache to be built beforehand under the
HighSpeedTrain cache root, for example:

    cache/<TrainType>/triplane_32x128x128_fp32/<case_name>.npz

TripNet geometry input intentionally uses only the triplane cache. It does not
use surface normals or area features. Flow condition is provided by the
high-speed TripNet dataset as cond=[Ux/100, Uy/20].
"""

CONFIG = {
    "dataset": "HighSpeedTrain",
    "dataset_option": "tripnet_volume",
    "model": "TripNet",
    "model_size": "MediumVolumeAll",
    "task": {
        "surface_input_list": ["triplane"],
        "surface_target_list": [],
        "volume_target_list": ["U", "p", "k", "omega", "nut"],
        "coefficient_target_list": [],
        "query_type_channel": False,
        "losses": [
            {"domain": "volume", "field": "U", "pred_slice": (0, 3), "weight": 1.0},
            {"domain": "volume", "field": "p", "pred_slice": (3, 4), "weight": 1.0},
            {"domain": "volume", "field": "k", "pred_slice": (4, 5), "weight": 1.0},
            {"domain": "volume", "field": "omega", "pred_slice": (5, 6), "weight": 1.0},
            {"domain": "volume", "field": "nut", "pred_slice": (6, 7), "weight": 1.0},
        ],
        "eval_fields": [
            {"domain": "volume", "field": "U", "pred_slice": (0, 3)},
            {"domain": "volume", "field": "p", "pred_slice": (3, 4)},
            {"domain": "volume", "field": "k", "pred_slice": (4, 5)},
            {"domain": "volume", "field": "omega", "pred_slice": (5, 6)},
            {"domain": "volume", "field": "nut", "pred_slice": (6, 7)},
        ],
        "denormalize": {
            "U": {"type": "physical", "dataset": "HighSpeedTrain"},
            "p": {"type": "physical", "dataset": "HighSpeedTrain"},
            "k": {"type": "physical", "dataset": "HighSpeedTrain"},
            "omega": {"type": "physical", "dataset": "HighSpeedTrain"},
            "nut": {"type": "physical", "dataset": "HighSpeedTrain"},
        },
    },
    "runtime": {
        "seed": 200099,
        "matmul_precision": "high",
        "amp": True,
        "compile": True,
        "distributed": {
            "gpu_num": 2,
            "start_gpu": 0,
            "port": 45328,
            "backend": "nccl",
        },
    },
    "train": {
        "dataset_kwargs": {
            "split": "train",
        },
        "batch_size": 4,
        "workers": 4,
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "total_epoch": 200,
        "resume_epoch": 0,
        "save_step": 50,
        "clip_grad": 0.1,
        "final_lr_ratio": 0.01,
        "warmup_iters": 5000,
        "log_interval": 20,
        "saveroot": "/data/home/zdhs0017/zoujunhong/model/savemodel/field/HighSpeedTrain_TripNet_volumeAll",
        "checkpoint": "",
        "route_map": None,
        "dry_run_iters": 0,
    },
    "test": {
        "dataset_kwargs": {
            "split": "test",
        },
        "checkpoint": "",
        "batch_size": 1,
        "workers": 4,
        "query_chunk_size": 50000,
        "max_batches": 0,
        "log_interval": 20,
    },
    "visualization": {
        "dataset_kwargs": {
            "split": "test",
        },
        "checkpoint": "",
        "batch_size": 1,
        "workers": 0,
        "case_name": "CRH450_geo_19_000002_Ux_98.79_Uy_16.29",
        "query_chunk_size": 50000,
        "max_points": 20000,
        "volume_use_vtk_slices": True,
        "volume_slice_planes": ("x", "y", "z"),
        "volume_slice_max_points": 200000,
        "output_dir": "",
    },
}

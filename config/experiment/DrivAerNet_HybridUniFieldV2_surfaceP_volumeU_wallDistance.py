COMMON_DATASET_KWARGS = {
    "num_surface_points": 32768,
    "num_surface_query_points": None,
    "num_query_points": 6144,
    "surface_input_list": ["xyz", "normal", "area"],
    "surface_target_list": ["p"],
    "volume_target_list": ["U", "wall_distance"],
    "wall_distance_target_transform": "sqrt",
}


CONFIG = {
    "dataset": "DrivAerNet",
    "dataset_option": "surface_volume",
    "model": "HybridUniFieldV2",
    "model_size": "Large",
    "task": {
        "surface_input_list": ["xyz", "normal", "area"],
        "surface_target_list": ["p"],
        "volume_target_list": ["U", "wall_distance"],
        "query_type_channel": False,
        "losses": [
            {"domain": "surface", "field": "p", "pred_slice": (0, 1), "weight": 1.0},
            {"domain": "volume", "field": "U", "pred_slice": (0, 3), "weight": 1.0},
            {"domain": "volume", "field": "wall_distance", "pred_slice": (3, 4), "weight": 1.0},
        ],
        "eval_fields": [
            {"domain": "surface", "field": "p", "pred_slice": (0, 1)},
            {"domain": "volume", "field": "U", "pred_slice": (0, 3)},
            {"domain": "volume", "field": "wall_distance", "pred_slice": (3, 4)},
        ],
        "denormalize": {
            "p": {"type": "affine", "mean": [-94.5], "std": [117.25]},
            "U": {
                "type": "affine",
                "mean": [21.725932072168813, -0.23861807530826643, 0.6522818340397506],
                "std": [12.373392759706936, 3.8865940380244175, 3.9738297581062536],
            },
        },
    },
    "runtime": {
        "seed": 200099,
        "matmul_precision": "high",
        "amp": True,
        "compile": False,
        "distributed": {
            "gpu_num": 1,
            "start_gpu": 0,
            "port": 45325,
            "backend": "nccl",
        },
    },
    "train": {
        "dataset_kwargs": {
            **COMMON_DATASET_KWARGS,
            "ids_file": "/data/group/project1/CFD/DrivAerNet++/train_design_ids.txt",
        },
        "batch_size": 1,
        "workers": 4,
        "lr": 1e-4,
        "weight_decay": 0.04,
        "total_epoch": 200,
        "resume_epoch": 0,
        "save_step": 20,
        "clip_grad": 0.1,
        "final_lr_ratio": 0.01,
        "warmup_iters": 5000,
        "log_interval": 10,
        "saveroot": (
            "/data/home/zdhs0017/zoujunhong/model/savemodel/field/"
            "DrivAerNet_HybridUniFieldV2_surfaceP_volumeU_wallDistance"
        ),
        "checkpoint": "",
        "route_map": None,
        "dry_run_iters": 0,
    },
    "test": {
        "dataset_kwargs": {
            **COMMON_DATASET_KWARGS,
            "ids_file": "/data/group/project1/CFD/DrivAerNet++/val_design_ids.txt",
        },
        "checkpoint": None,
        "batch_size": 1,
        "workers": 4,
        "query_chunk_size": 50000,
        "max_batches": 0,
        "log_interval": 20,
    },
    "visualization": {
        "dataset_kwargs": {
            **COMMON_DATASET_KWARGS,
            "ids_file": "/data/group/project1/CFD/DrivAerNet++/val_design_ids.txt",
        },
        "checkpoint": None,
        "batch_size": 1,
        "workers": 0,
        "query_chunk_size": 50000,
        "max_points": 20000,
        "output_dir": "",
    },
}

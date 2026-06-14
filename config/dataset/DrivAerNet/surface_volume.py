CONFIG = {
    "name": "DrivAerNet",
    "kwargs": {
        "use_surface": True,
        "use_volume": True,
        "num_surface_points": 8192,
        "num_surface_query_points": 2048,
        "num_query_points": 6144,
        "surface_input_list": ["xyz", "normal", "area"],
        "surface_target_list": ["p"],
        "volume_target_list": ["U"],
        "surface_sampling": "random",
        "normalization": "standard",
        "ids_file": "/data/group/project1/CFD/DrivAerNet++/train_design_ids.txt",
        "repeat": 1,
        "route": 0,
    },
}

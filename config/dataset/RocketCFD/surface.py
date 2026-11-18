"""RocketCFD surface-Cp dataset with explicit five-way split selection."""

CONFIG = {
    "name": "RocketCFD",
    "kwargs": {
        "use_surface": True,
        "use_volume": False,
        "num_surface_points": 8192,
        # Supervise the exact encoder points and bypass decoder-side KNN.
        "num_surface_query_points": None,
        "num_query_points": 0,
        "surface_input_list": ["xyz"],
        "surface_target_list": ["p"],
        "volume_target_list": [],
        "coefficient_target_list": [],
        "surface_sampling": "random",
        "normalization": "physical",
        "condition_representation": "canonical",
        "rotate_geometry_by_aoa": True,
        "ground_z": False,
        "subset": "train",
        "quality_policy": "all_completed",
        "excluded_geometry_ids": ("g020", "g050"),
        "split": "all",
        "skip_incomplete": True,
        "repeat": 1,
        "route": 0,
    },
}

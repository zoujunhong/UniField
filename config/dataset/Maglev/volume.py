CONFIG = {
    "name": "Maglev",
    "kwargs": {
        "use_surface": False,
        "use_volume": True,
        "num_surface_points": 0,
        "num_surface_query_points": 0,
        "num_query_points": 8192,
        "surface_input_list": ["xyz", "normal", "area"],
        "surface_target_list": [],
        "volume_target_list": ["U"],
        "surface_sampling": "random",
        "normalization": "physical",
        "split": "train",
        "repeat": 1,
        "route": 2,
    },
}

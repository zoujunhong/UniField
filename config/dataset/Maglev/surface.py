CONFIG = {
    "name": "Maglev",
    "kwargs": {
        "use_surface": True,
        "use_volume": False,
        "num_surface_points": 8192,
        "num_surface_query_points": 2048,
        "num_query_points": 0,
        "surface_input_list": ["xyz", "normal", "area"],
        "surface_target_list": ["p"],
        "volume_target_list": [],
        "surface_sampling": "random",
        "normalization": "physical",
        "split": "train",
        "repeat": 1,
        "route": 2,
    },
}

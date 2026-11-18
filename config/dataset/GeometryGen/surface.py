"""Canonical GeometryGen surface-Cp cache with FPS inputs and random queries."""

CONFIG = {
    "name": "GeometryGenSurface",
    "kwargs": {
        "use_surface": True,
        "use_volume": False,
        "num_surface_points": 8192,
        "num_surface_query_points": 8192,
        "num_query_points": 0,
        "surface_input_list": ["xyz", "normal"],
        # The shared task interface names the scalar pressure target `p`; this
        # dataset stores dimensionless Cp in that slot.
        "surface_target_list": ["p"],
        "volume_target_list": [],
        "split": "train",
        "val_ratio": 0.1,
        "exclude_ids_file": "config/dataset/GeometryGen/excluded_cp_outliers.txt",
        "repeat": 1,
        "route": 0,
    },
}

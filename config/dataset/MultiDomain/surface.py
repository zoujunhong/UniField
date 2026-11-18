"""Canonical five-domain surface-pressure training configuration.

``selected_domains`` can be overridden per train/test/visualization phase with
``dataset_kwargs``.  Routes distinguish the five physical application domains;
every sample uses ``[U_inf / 100, placement]`` and the canonical geometry frame.
"""

CONFIG = {
    "name": [
        "DrivAerNet",
        "CRH450",
        "Maglev",
        "GeneratedHighSpeedTrain",
        "AircraftWing",
        "FullAircraft",
        "ThingiCFD",
    ],
    "domain_groups": {
        "automobile": ["DrivAerNet"],
        "high_speed_train": ["CRH450", "Maglev", "GeneratedHighSpeedTrain"],
        "aircraft_wing": ["AircraftWing"],
        "full_aircraft": ["FullAircraft"],
        "generic_shape": ["ThingiCFD"],
    },
    "selected_domains": [
        "automobile",
        "high_speed_train",
        "aircraft_wing",
        "full_aircraft",
        "generic_shape",
    ],
    "common_kwargs": {
        "use_surface": True,
        "use_volume": False,
        "num_surface_points": 8192,
        "num_surface_query_points": 2048,
        "num_query_points": 0,
        "surface_input_list": ["xyz", "normal", "area"],
        "surface_target_list": ["p"],
        "volume_target_list": [],
        "coefficient_target_list": [],
        "surface_sampling": "random",
        "normalization": "physical",
        "repeat": 1,
    },
    "kwargs": {
        "DrivAerNet": {
            "cond": [0.3, 0.0],
            "reynolds_similarity_speeds": list(range(5, 101, 5)),
            "reynolds_similarity_reference_speed": 30.0,
            "route": 0,
        },
        "CRH450": {"align_freestream": True, "route": 1},
        "Maglev": {"align_freestream": True, "route": 1},
        "GeneratedHighSpeedTrain": {"align_freestream": True, "route": 1},
        "AircraftWing": {"condition_representation": "canonical", "rotate_geometry_by_aoa": True, "route": 2},
        "FullAircraft": {"condition_representation": "canonical", "rotate_geometry_by_aoa": True, "route": 3},
        "ThingiCFD": {"route": 4},
    },
}

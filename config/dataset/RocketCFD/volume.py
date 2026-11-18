CONFIG = {
    'name': 'RocketSurfaceVolume',
    'kwargs': {
        'split': 'train', 'num_surface_points': 8192,
        'num_surface_query_points': 2048, 'num_query_points': 8192,
        'surface_input_list': ['xyz', 'normal'],
        'surface_target_list': ['p', 'surface_distance'],
        'volume_target_list': ['U', 'p', 'surface_distance'],
        'coefficient_target_list': [],
    },
}

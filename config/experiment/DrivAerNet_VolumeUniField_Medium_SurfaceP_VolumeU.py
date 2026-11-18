from config.defaults import make_experiment

CONFIG = make_experiment('DrivAerNet', 'VolumeUniField', 'Medium', surface=('p',), volume=('U',), dataset_option='surface_volume')
CONFIG["experiment"]["name"] = 'DrivAerNet_VolumeUniField_Medium_SurfaceP_VolumeU'

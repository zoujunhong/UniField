from config.defaults import make_experiment

CONFIG = make_experiment('DrivAerNet', 'UniField', 'Large', surface=('p',), volume=('U', 'surface_distance'), dataset_option='surface_volume')
CONFIG["experiment"]["name"] = 'DrivAerNet_UniField_Large_SurfaceP_VolumeUDistance'

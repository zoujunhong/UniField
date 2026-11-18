from config.defaults import make_experiment

CONFIG = make_experiment('AircraftWing', 'UniField', 'Large', surface=('p',), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'Wing_UniField_Large_SurfaceP_VolumeX'

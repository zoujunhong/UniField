from config.defaults import make_experiment

CONFIG = make_experiment('FullAircraft', 'UniField', 'Large', surface=('p',), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'Aircraft_UniField_Large_SurfaceP_VolumeX'

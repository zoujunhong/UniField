from config.defaults import make_experiment

CONFIG = make_experiment('ThingiCFD', 'UniField', 'Large', surface=('p',), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'ThingiCFD_UniField_Large_SurfaceP_VolumeX'

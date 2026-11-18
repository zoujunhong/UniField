from config.defaults import make_experiment

CONFIG = make_experiment('HighSpeedTrain', 'UniField', 'Large', surface=('p',), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'HighSpeedTrain_UniField_Large_SurfaceP_VolumeX'

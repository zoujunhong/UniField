from config.defaults import make_experiment

CONFIG = make_experiment('DrivAerNet', 'UniField', 'Large', surface=('p', 'normal'), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'DrivAerNet_UniField_Large_SurfacePNormal_VolumeX'

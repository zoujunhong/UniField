from config.defaults import make_experiment

CONFIG = make_experiment('HighSpeedTrain', 'VolumeUniField', 'Medium', surface=(), volume=('p', 'U', 'k', 'omega', 'nut'), dataset_option='volume')
CONFIG["experiment"]["name"] = 'HighSpeedTrain_VolumeUniField_Medium_SurfaceX_VolumePUKOmegaNut'

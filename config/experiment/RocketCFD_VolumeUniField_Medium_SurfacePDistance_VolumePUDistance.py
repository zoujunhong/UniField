from config.defaults import make_experiment

CONFIG = make_experiment('RocketCFD', 'VolumeUniField', 'Medium', surface=('p', 'surface_distance'), volume=('p', 'U', 'surface_distance'), dataset_option='volume')
CONFIG["experiment"]["name"] = 'RocketCFD_VolumeUniField_Medium_SurfacePDistance_VolumePUDistance'
CONFIG["dataset_kwargs"].pop("use_volume", None)

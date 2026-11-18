from config.defaults import make_experiment

CONFIG = make_experiment('GeometryGen', 'VolumeUniField', 'Medium', surface=('p', 'U', 'surface_distance'), volume=('p', 'U', 'surface_distance'), dataset_option='volume')
CONFIG["experiment"]["name"] = 'GeometryGen_VolumeUniField_Medium_SurfacePUDistance_VolumePUDistance'
CONFIG["dataset_kwargs"].pop("use_volume", None)
CONFIG["dataset_kwargs"]["split_file"] = '/data/home/zdhs0017/zoujunhong/code/whole_field/dataset/splits/GeometryGen/quality13192_val1500_seed20261027.json'
CONFIG["test"]["dataset_kwargs"]["split"] = "val"
CONFIG["train"]["auto_test"] = False  # No independent frozen test split in this historical dataset.

from config.defaults import make_experiment

CONFIG = make_experiment('RocketCFD', 'UniField', 'Large', surface=('p',), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'RocketCFD_UniField_Large_SurfaceP_VolumeX'
CONFIG["train"]["dataset_kwargs"] = {"subset": "train"}
CONFIG["validation"] = {"dataset_kwargs": {"subset": "val"}}
CONFIG["test"]["dataset_kwargs"] = {"subset": "test", "deterministic": True}

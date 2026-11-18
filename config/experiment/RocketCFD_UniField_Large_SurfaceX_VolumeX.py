from config.defaults import make_experiment

CONFIG = make_experiment('RocketCFD', 'UniField', 'Large', surface=(), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'RocketCFD_UniField_Large_SurfaceX_VolumeX'
CONFIG["train"]["dataset_kwargs"] = {"subset": "train"}
CONFIG["validation"] = {"dataset_kwargs": {"subset": "val"}}
CONFIG["test"]["dataset_kwargs"] = {"subset": "test", "deterministic": True}
CONFIG["model"].pop("decoder_name", None)
CONFIG["model"].pop("decoder_kwargs", None)
CONFIG["model"]["coefficient_head_kwargs"] = {"out_channels": 2, "hidden_dim": 1024, "depth": 2}
CONFIG["task"]["coefficient_target_list"] = ["Cd", "Cl"]
CONFIG["dataset_kwargs"]["coefficient_target_list"] = ["Cd", "Cl"]
CONFIG["task"]["losses"] = [{"domain":"coefficients", "field":f, "pred_slice":(i,i+1), "loss_type":"mse", "weight":1.0} for i,f in enumerate(["Cd","Cl"])]
CONFIG["task"]["eval_fields"] = [{"domain":"coefficients", "field":f, "pred_slice":(i,i+1)} for i,f in enumerate(["Cd","Cl"])]

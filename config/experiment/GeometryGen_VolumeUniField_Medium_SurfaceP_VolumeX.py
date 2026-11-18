from config.defaults import make_experiment

CONFIG = make_experiment('GeometryGen', 'VolumeUniField', 'Medium', surface=('p',), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'GeometryGen_VolumeUniField_Medium_SurfaceP_VolumeX'

# Canonical surface cache stores a 3D body-frame velocity condition.
CONFIG['model']['encoder_kwargs']['cond_dim'] = 3
CONFIG['model']['decoder_kwargs']['cond_dim'] = 3
CONFIG['train']['auto_test'] = False  # This historical dataset has only train/validation.

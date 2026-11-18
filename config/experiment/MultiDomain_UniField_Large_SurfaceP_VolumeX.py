from config.defaults import make_experiment

CONFIG = make_experiment('MultiDomain', 'UniField', 'Large', surface=('p',), volume=(), dataset_option='surface')
CONFIG["experiment"]["name"] = 'MultiDomain_UniField_Large_SurfaceP_VolumeX'
CONFIG["model"]["encoder_kwargs"]["num_routes"] = 5
CONFIG["model"]["decoder_kwargs"]["num_routes"] = 5

CONFIG['dataset_kwargs']['DrivAerNet'] = {'validity_file': str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'dataset/splits/DrivAerNet/validity.json')}
CONFIG['test']['dataset_kwargs']['DrivAerNet'] = {
    'reynolds_similarity_speeds': None,
    'reynolds_similarity_factor_range': None,
}

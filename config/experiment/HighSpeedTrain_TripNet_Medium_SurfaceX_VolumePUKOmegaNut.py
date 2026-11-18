from config.defaults import make_experiment

CONFIG = make_experiment('HighSpeedTrain', 'TripNet', 'Medium', surface=(), volume=('p','U','k','omega','nut'), dataset_option='tripnet_volume')
CONFIG['task']['surface_input_list'] = ['triplane']
CONFIG['dataset_kwargs']['surface_input_list'] = ['triplane']

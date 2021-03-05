import argparse
import json
import os
import glob

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from Dataset import *
from Layers import *
from Models import *
from Losses import *
from Metrics import *
from Utils import *
from PyFire import Trainer
from VisualizationsAndDemonstrations import *

if __name__ == '__main__':
	print('Running Experiment')
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--animal', type=str,
						help='animal root directory')
	parser.add_argument('-c', '--config', type=str,
						help='JSON file for configuration')
	parser.add_argument('-d', '--data', type=str,
						help='data directory')
	args = parser.parse_args()

	root = args.animal
	if root[-1] != r'/':
		root += r'/'

	with open(root + args.config) as f:
		data = f.read()
	config = json.loads(data)

	global classifier_dataset_config
	classifier_dataset_config = config['classifier_dataset_params']
	global classifier_learning_params
	classifier_learning_params = config['classifier_learning_params']
	global classifier_trainer_params
	classifier_trainer_params = config['classifier_trainer_params']
	global classifier_model_config
	classifier_model_config = config['classifier_model_params']

	global separator_dataset_config
	separator_dataset_config = config['separator_dataset_params']
	global separator_learning_params
	separator_learning_params = config['separator_learning_params']
	global separator_trainer_params
	separator_trainer_params = config['separator_trainer_params']
	global separator_model_config
	separator_model_config = config['separator_model_params']
        
	print('Input Mode:', separator_model_config['input_mode'])
	print('Output Mode:', separator_model_config['output_mode'])
	print('Phase Channel:', separator_model_config['phase_channel'])
	print('Objective:', separator_trainer_params['loss_func'])

	nfft = separator_model_config['filterbank_params']['nfft']
	hop = separator_model_config['filterbank_params']['hop']
	stft = STFT(nfft, hop, dB=False)
	stft_db = STFT(nfft, hop, dB=True)
	losses_dict = {
		'nll':F.nll_loss,
		'nll_weighted': lambda x,y: F.nll_loss(x, y, nll_weights.to(classifier_trainer_params['device'])),
		'mae':mae_loss,
		'mse':mse_loss,
		'r2s_mae':lambda x,y: raw2spec_mae_loss(x, y, stft),
		'r2s_mse':lambda x,y:raw2spec_mse_loss(x, y, stft),
		'r2sdb_mae':lambda x,y:raw2spec_mae_loss(x, y, stft_db),
		'r2sdb_mse':lambda x,y:raw2spec_mse_loss(x, y, stft_db),
		'spec_conv':lambda x,y:spectral_convergence_loss(x, y),
		'r2s_spec_conv':lambda x,y:raw2spec_spectral_convergence_loss(x, y, stft),
		'nsisdr':neg_si_sdr,
		'total':lambda x,y:total_loss(x, y, stft),
		'pit_mae':pit_mae_loss,
		'pit_mse':pit_mse_loss,
		'pit_r2s_mae':lambda x,y:pit_raw2spec_mae_loss(x, y, stft),
		'pit_r2s_mse':lambda x,y:pit_raw2spec_mse_loss(x, y, stft),
		'pit_r2sdb_mae':lambda x,y:pit_raw2spec_mae_loss(x, y, stft_db),
		'pit_r2sdb_mse':lambda x,y:pit_raw2spec_mse_loss(x, y, stft_db),
		'pit_spec_conv':lambda x,y:pit_spectral_convergence_loss(x, y),
		'pit_r2s_spec_conv':lambda x,y:pit_raw2spec_spectral_convergence_loss(x, y, stft),
		'pit_nsisdr':pit_neg_si_sdr,
		'pit_total':lambda x,y:pit_total_loss(x, y, stft)
	}
	metrics_dict = lambda clsfr: {
		'classifier_acc':accuracy,
		'sisdr':si_sdr,
		'separator_acc':lambda x,y: accuracy(x, y, index=2, classifier=clsfr),
		'pit_sisdr':lambda x,y:pit_si_sdr(x, y, 1),
		'pit_separator_acc':lambda x,y: pit_accuracy(x, y, index=2, classifier=clsfr)
	}

	X_train = torch.load(root+f'{args.data}/X_train.pt')
	Y_train = torch.load(root+f'{args.data}/Y_train.pt')
	Y_train_id = torch.load(root+f'{args.data}/Y_train_id.pt')
	X_test = torch.load(root+f'{args.data}/X_test.pt')
	Y_test = torch.load(root+f'{args.data}/Y_test.pt')
	Y_test_id = torch.load(root+f'{args.data}/Y_test_id.pt')

	if not os.path.isdir(root+'Classifier'):
		classifier_trainer_params['dest'] = root+'Classifier'
		assert classifier_trainer_params['dest'] == root+'Classifier'

		classifier_trainer_params['loss_func'][list(classifier_trainer_params['loss_func'].keys())[0]] = \
					losses_dict[classifier_trainer_params['loss_func'][list(classifier_trainer_params['loss_func'].keys())[0]]]
		classifier_trainer_params['metric_func'][list(classifier_trainer_params['metric_func'].keys())[0]] = \
					metrics_dict(None)[classifier_trainer_params['metric_func'][list(classifier_trainer_params['metric_func'].keys())[0]]]
		
		classifier_dataset_train = PipelineDataset(X_train, 
												   Y_train, 
												   Y_train_id,
												   **classifier_dataset_config)

		classifier_dataset_test = PipelineDataset(X_test, 
												  Y_test, 
												  Y_test_id,
												  **classifier_dataset_config)

		classifier_dataloader_train = torch.utils.data.DataLoader(classifier_dataset_train,
																  batch_size=classifier_learning_params['batch_size'],
																  shuffle=True)
		classifier_dataloader_test = torch.utils.data.DataLoader(classifier_dataset_test,
																 batch_size=classifier_learning_params['batch_size'],
																 shuffle=False)

		model = Classifier(**classifier_model_config)
		optimizer = optim.Adam(model.parameters(), lr=classifier_learning_params['learning_rate'])
		trainer = Trainer(model, optimizer, **classifier_trainer_params)
		trainer.fit(classifier_dataloader_train, classifier_dataloader_test, classifier_learning_params['epochs'])
		trainer.save_model()

		classifier_data_train, classifier_predictions_train = trainer.evaluate(classifier_dataloader_train, 'train', to_device='cuda')
		classifier_data_test, classifier_predictions_test = trainer.evaluate(classifier_dataloader_test, 'test', to_device='cuda')

	else:
		classifier_trainer_params['dest'] = None
		assert classifier_trainer_params['dest'] == None

	classifier_path = glob.glob(root+'Classifier/Models/*.pt')[0]

	classifier = Classifier(**classifier_model_config)
	classifier.load_state_dict(torch.load(classifier_path))
	classifier.eval()

	separator_dataset_train = PipelineDataset(X_train, 
											  Y_train, 
											  Y_train_id, 
											  **separator_dataset_config)
	separator_dataset_test = PipelineDataset(X_test, 
											 Y_test, 
											 Y_test_id, 
											 **separator_dataset_config)

	separator_dataloader_train = torch.utils.data.DataLoader(separator_dataset_train,
															 batch_size=separator_learning_params['batch_size'],
															 shuffle=True)
	separator_dataloader_test = torch.utils.data.DataLoader(separator_dataset_test,
															batch_size=separator_learning_params['batch_size'],
															shuffle=False)

	separator_trainer_params['loss_func'][list(separator_trainer_params['loss_func'].keys())[0]] = \
					losses_dict[separator_trainer_params['loss_func'][list(separator_trainer_params['loss_func'].keys())[0]]]
	separator_trainer_params['metric_func'][list(separator_trainer_params['metric_func'].keys())[0]] = \
					metrics_dict(classifier)[separator_trainer_params['metric_func'][list(separator_trainer_params['metric_func'].keys())[0]]]

	for k in separator_trainer_params['params'].keys():
		if 'loss' in k:
			separator_trainer_params['params'][k] = losses_dict[separator_trainer_params['params'][k]]
		elif 'metric' in k:
			separator_trainer_params['params'][k] = metrics_dict(classifier)[separator_trainer_params['params'][k]]
	model = RepUNet(**separator_model_config)
	model.apply(weights_init)

	try:
		opt = separator_learning_params['optimizer']
	except KeyError:
		opt = None

	if opt == 'adamw':
		optimizer = optim.AdamW(model.parameters(), lr=separator_learning_params['learning_rate'])
	elif opt == 'adamw_amsgrad':
		optimizer = optim.AdamW(model.parameters(), 
								lr=separator_learning_params['learning_rate'],
								amsgrad=True)
	elif opt == 'sgd':
		optimizer = optim.SGD(model.parameters(),
							  lr=separator_learning_params['learning_rate'],
							  momentum=separator_learning_params['momentum'],
							  nesterov=True)
	else:
		optimizer = optim.AdamW(model.parameters(), lr=separator_learning_params['learning_rate'])

	try:
		switcher = separator_trainer_params['params']['optimizer_switcher_callback']
		if switcher['optimizer'] == 'adamw':
			switcher['optimizer'] = lambda model: optim.AdamW(model.parameters(), lr=switcher['learning_rate'])
		elif switcher['optimizer'] == 'adamw_amsgrad':
			switcher['optimizer'] = lambda model: optim.AdamW(model.parameters(), 
												lr=switcher['learning_rate'],
												amsgrad=True)
		separator_trainer_params['params']['optimizer_switcher_callback'] = switcher
	except KeyError:
		switcher = None

	trainer = Trainer(model, optimizer, 
					  loss_func=separator_trainer_params['loss_func'],
					  metric_func=separator_trainer_params['metric_func'],
					  verbose=separator_trainer_params['verbose'],
					  device=separator_trainer_params['device'],
					  dest=root+separator_trainer_params['dest'],
					  **separator_trainer_params['params'])
	trainer.fit(separator_dataloader_train, separator_dataloader_test, separator_learning_params['epochs'])
    
	try:
		eval_return = config['eval_return_data']
	except KeyError:
		eval_return = True
        
	separator_data_train, separator_predictions_train = trainer.evaluate(separator_dataloader_train, 
																		 'train', 
																		 to_device='cuda',
																		 return_data=eval_return)
	separator_data_test, separator_predictions_test = trainer.evaluate(separator_dataloader_test, 
																	   'test', 
																	   to_device='cuda',
																	   return_data=eval_return)
	trainer.save_model()
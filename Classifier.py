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

	X_train = torch.load(root+f'{args.data}/Classifier/X_train.pt')
	Y_train = torch.load(root+f'{args.data}/Classifier/Y_train.pt')

	X_test = torch.load(root+f'{args.data}/Classifier/X_test.pt')
	Y_test = torch.load(root+f'{args.data}/Classifier/Y_test.pt')

	Y_train = id_mapper(Y_train)
	Y_test = id_mapper(Y_test)

	nll_weights = torch.Tensor(nll_loss_weights(Y_train.numpy()))

	nfft = classifier_model_config['stft_params']['kernel_size']
	hop = classifier_model_config['stft_params']['stride']
	stft = STFT(nfft, hop, dB=False)
	stft_db = STFT(nfft, hop, dB=True)
	losses_dict = {
		'nll':F.nll_loss,
		'nll_weighted': lambda x,y: F.nll_loss(x, y, nll_weights.to(classifier_trainer_params['device'])),
		'mae':mae_loss,
		'mse':mse_loss,
		'r2s_mae':lambda x,y:raw2spec_mae_loss(x, y, stft),
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
		'pit_separator_acc':lambda x,y: pit_accuracy(x, y, index=2, classifier=clsfr),
		'pit_probnorm_acc':lambda x,y: pit_probnorm_accuracy(x, y, index=2, classifier=clsfr, peak_accuracy=None),
	}

	if not os.path.isdir(root+classifier_trainer_params['dest']):
		classifier_trainer_params['dest'] = root+classifier_trainer_params['dest']

		classifier_trainer_params['loss_func'][list(classifier_trainer_params['loss_func'].keys())[0]] = \
					losses_dict[classifier_trainer_params['loss_func'][list(classifier_trainer_params['loss_func'].keys())[0]]]
		classifier_trainer_params['metric_func'][list(classifier_trainer_params['metric_func'].keys())[0]] = \
					metrics_dict(None)[classifier_trainer_params['metric_func'][list(classifier_trainer_params['metric_func'].keys())[0]]]
		
		classifier_dataset_train = ClassifierDataset(X_train, 
													 Y_train)

		classifier_dataset_test = ClassifierDataset(X_test, 
													Y_test)

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

		try:
			eval_return = config['eval_return_data']
		except KeyError:
			eval_return = True

		classifier_data_train, classifier_predictions_train = trainer.evaluate(classifier_dataloader_train, 
																			   'train', 
																			   to_device='cuda',
																			   return_data=eval_return)
		classifier_data_test, classifier_predictions_test = trainer.evaluate(classifier_dataloader_test, 
																			 'test', 
																			 to_device='cuda',
																			 return_data=eval_return)
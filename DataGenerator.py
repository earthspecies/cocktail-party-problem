import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys
import json
import random
from sklearn.model_selection import train_test_split

from DataHelpers import *
from DataMixer import *
from Dataset import *
from Layers import *
from Models import *
from Losses import *
from Metrics import *
from Utils import *
from PyFire import Trainer
from VisualizationsAndDemonstrations import *

def str_to_bool(value):
	if value.lower() in {'false', 'f', '0', 'no', 'n'}:
		return False
	elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
		return True
	raise ValueError(f'{value} is not a valid boolean value')

def generate_labeled_waveforms(animal, *args, **kwargs):
	if animal == 'Macaque':
		loader = LoadMacaqueData(os=kwargs['os'])
		X, Y = loader.run(balance=kwargs['balance'])
	elif animal == 'Dolphin':
		loader = LoadDolphinData(os=kwargs['os'], n_individuals=kwargs['n_individuals'])
		X, Y = loader.run()
	elif animal == 'Bat':
		loader = LoadBatData(os=kwargs['os'])
		X, Y = loader.run(balance=kwargs['balance'])
	elif animal == 'SpermWhale':
		loader = LoadSpermWhaleData(os=kwargs['os'])
		X, Y = loader.run(balance=kwargs['balance'])        
	return X, Y

def open_closed_split(X, Y, n_open=None, seed=42):
	random.seed(seed)

	ids = np.unique(Y).tolist()
	if n_open is not None:
		open_ids = random.sample(list(np.unique(Y)), n_open)
	else:
		open_ids = []
	closed_ids = [id for id in ids if id not in open_ids]
	
	X_closed = [x for x,y in zip(X, Y) if y in closed_ids]
	Y_closed = [y for y in Y if y in closed_ids]

	if n_open is not None:
		X_open = [x for x,y in zip(X, Y) if y in open_ids]
		Y_open = [y for y in Y if y in open_ids]
	else:
		X_open, Y_open = None, None

	return X_closed, Y_closed, X_open, Y_open

def generate_classifier_data(X, Y, *args, **kwargs):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
														test_size=0.2, 
														random_state=42)

	aug_factor = kwargs['augmentation_factor']
	if aug_factor is not None:
		X_train, Y_train = augmenter(X_train, Y_train,
									 augmentation_factor=aug_factor,
									 shift_factor=kwargs['shift_factor'],
									 pad=kwargs['padding_scheme'],
									 side=kwargs['padding_side'],
									 shuffle=True)

		X_test, Y_test = augmenter(X_test, Y_test,
								   augmentation_factor=aug_factor,
								   shift_factor=kwargs['shift_factor'],
								   pad=kwargs['padding_scheme'],
								   side=kwargs['padding_side'],
								   shuffle=True)

	X_train = torch.Tensor(X_train)
	X_test = torch.Tensor(X_test)
	Y_train = torch.LongTensor(Y_train)
	Y_test = torch.LongTensor(Y_test)

	return X_train, X_test, Y_train, Y_test

def generate_mixture_data(X, Y, *args, **kwargs):
	mixer = SourceMixer(n_src=kwargs['n_src'], 
						samples=kwargs['n_samples'], 
						frames=X[0].shape[-1])

	x_mix, y_mix, y_mix_id = mixer.mix(X, 
									   Y, 
									   shift_factor=kwargs['shift_factor'], 
									   shift_overlaps=kwargs['shift_overlaps'],
									   pad=kwargs['padding_scheme'],
									   side=kwargs['padding_side'])
	X = None
	Y = None
	del X
	del Y

	x_mix = torch.stack(x_mix, dim=0)
	y_mix = torch.stack(y_mix, dim=0)
	y_mix_id = torch.stack(y_mix_id, dim=0)
	return x_mix, y_mix, y_mix_id

def train_val_split(X, Y, seed, *args, **kwargs):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
														test_size=0.2, 
														random_state=seed)

	aug_factor = kwargs['augmentation_factor']
	if aug_factor is not None:
		X_train, Y_train = augmenter(X_train, Y_train,
									 augmentation_factor=aug_factor,
									 shift_factor=kwargs['shift_factor'],
									 pad=kwargs['padding_scheme'],
									 side=kwargs['padding_side'],
									 shuffle=True)

		X_test, Y_test = augmenter(X_test, Y_test,
								   augmentation_factor=aug_factor,
								   shift_factor=kwargs['shift_factor'],
								   pad=kwargs['padding_scheme'],
								   side=kwargs['padding_side'],
								   shuffle=True)

	return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser = argparse.ArgumentParser()
	parser.add_argument('--os', type=str,
						help='specify Windows or Ubuntu', 
						default='Ubuntu')
	parser.add_argument('--data_directory', type=str,
						help='root directory for data files, e.g. Data')
	parser.add_argument('--animal', type=str,
						help='specify the animal to analyze')
	parser.add_argument('--config', type=str,
						help='config.json file name')
	parser.add_argument('--objective', type=str,
						help='objective task, e.g. Classification or Separation')
	parser.add_argument('--regime', type=str,
						help='open or closed regime, e.g. Open or Closed')
	parser.add_argument('--seed', type=int,
						default=42)

	args = parser.parse_args()
	
	animal = args.animal
	assert animal in ['Macaque', 'Dolphin', 'Bat', 'SpermWhale']
	if not os.path.isdir(animal):
		os.mkdir(animal)

	with open(animal + r'/' + args.config) as f:
		data = f.read()
	config = json.loads(data)

	global preprocessing_config
	preprocessing_config = config['data_preprocessing']
	general_preprocessing = preprocessing_config['general']
	classifier_preprocessing = preprocessing_config['classifier']
	separator_preprocessing = preprocessing_config['separator']

	n_src=general_preprocessing['n_src']

	root = args.data_directory
	if root[-1] != r'/':
		root += r'/'
	root = animal + '/' + root
	if not os.path.isdir(root):
		os.mkdir(root)
	
	random.seed(args.seed)
	np.random.seed(args.seed)

	task = args.objective
	assert task in ['Classification', 'Separation']

	regime = args.regime
	assert regime in ['Open', 'Closed']

	if not os.path.isdir(root+'Waveforms'):
		waveforms_dir = 'Waveforms/'
		os.mkdir(root + waveforms_dir)
		X, Y = generate_labeled_waveforms(animal,
										  os=args.os,
										  balance=general_preprocessing['balance'],
										  n_individuals=general_preprocessing['n_individuals'])
		X_closed, Y_closed, X_open, Y_open = open_closed_split(X, Y,
															   n_open=general_preprocessing['n_open'],
															   seed=args.seed)
			
		
		np.save(root+f'{waveforms_dir}X_closed.npy', X_closed)
		np.save(root+f'{waveforms_dir}Y_closed.npy', Y_closed)
		if X_open is not None:
			np.save(root+f'{waveforms_dir}X_open.npy', X_open)
			np.save(root+f'{waveforms_dir}Y_open.npy', Y_open)
	else:
		X_closed = [row for row in np.load(root+'Waveforms/X_closed.npy')]
		Y_closed = [y for y in np.load(root+'Waveforms/Y_closed.npy')]

		try:
			X_open = [row for row in np.load(root+'Waveforms/X_open.npy')]
			Y_open = [y for y in np.load(root+'Waveforms/Y_open.npy')]
		except FileNotFoundError:
			assert regime=='Closed', print('Open regime data cannot be found')

	zipfile_save = general_preprocessing['zipfile_save']            
	if task == 'Classification':
		X_open, Y_open = None, None
		del X_open
		del Y_open
		
		X_train, X_test, Y_train, Y_test = train_val_split(X_closed, Y_closed, seed=0,
														   augmentation_factor=classifier_preprocessing['augmentation_factor'],
														   shift_factor=classifier_preprocessing['shift_factor'],
														   padding_scheme=classifier_preprocessing['padding_scheme'],
														   padding_side=classifier_preprocessing['padding_side'])

		X_train = torch.Tensor(X_train)
		X_test = torch.Tensor(X_test)
		Y_train = torch.LongTensor(Y_train)
		Y_test = torch.LongTensor(Y_test)
		
		classifier_root = root + 'Classifier/'
		if not os.path.isdir(classifier_root):
			os.mkdir(classifier_root)

		torch.save(X_train.clone().detach(), classifier_root+'X_train.pt', _use_new_zipfile_serialization=zipfile_save)
		torch.save(Y_train.clone().detach(), classifier_root+'Y_train.pt', _use_new_zipfile_serialization=zipfile_save)
		torch.save(X_test.clone().detach(), classifier_root+'X_test.pt', _use_new_zipfile_serialization=zipfile_save)
		torch.save(Y_test.clone().detach(), classifier_root+'Y_test.pt', _use_new_zipfile_serialization=zipfile_save)

		X_train, Y_train, X_test, Y_test = None, None, None, None
		del X_train
		del Y_train
		del X_test
		del Y_test

	else:
		if regime == 'Closed':

			X_open, Y_open = None, None
			del X_open
			del Y_open

			X_train, X_test, Y_train, Y_test = train_val_split(X_closed, Y_closed, seed=args.seed,
															   augmentation_factor=None)

			X_closed, Y_closed = None, None
			del X_closed
			del Y_closed

			if separator_preprocessing['mixing_to_memory'] is not None:
				
				mixing_params = separator_preprocessing['mixing_to_memory']

				X_train, Y_train, Y_train_id = generate_mixture_data(X_train, Y_train,
																	 n_src=mixing_params['n_src'],
																	 n_samples=mixing_params['training_size'],
																	 shift_factor=mixing_params['shift_factor'],
																	 shift_overlaps=mixing_params['shift_overlaps'],
																	 padding_scheme=mixing_params['padding_scheme'],
																	 padding_side=mixing_params['padding_side'])
                
				task_directory = f'SeparatorClosed{n_src}Speakers/'

				task_root = root + task_directory
				if not os.path.isdir(task_root):
					os.mkdir(task_root)
                
				torch.save(X_train, task_root + 'X_train.pt')
				torch.save(Y_train, task_root + 'Y_train.pt')
				torch.save(Y_train_id, task_root + 'Y_train_id.pt')

				X_train, Y_train, Y_train_id = None, None, None
				del X_train
				del Y_train
				del Y_train_id

				X_test, Y_test, Y_test_id = generate_mixture_data(X_test, Y_test,
																  n_src=mixing_params['n_src'],
																  n_samples=mixing_params['validation_size'],
																  shift_factor=mixing_params['shift_factor'],
																  shift_overlaps=mixing_params['shift_overlaps'],
																  padding_scheme=mixing_params['padding_scheme'],
																  padding_side=mixing_params['padding_side'])

				torch.save(X_test, task_root + 'X_test.pt')
				torch.save(Y_test, task_root + 'Y_test.pt')
				torch.save(Y_test_id, task_root + 'Y_test_id.pt')

				X_test, Y_test, Y_test_id = None, None, None
				del X_test
				del Y_test
				del Y_test_id

			else:
                
				task_directory = f'SeparatorClosed/'

				task_root = root + task_directory
				if not os.path.isdir(task_root):
					os.mkdir(task_root)
                
				X_train = torch.Tensor(X_train)
				X_test = torch.Tensor(X_test)
				Y_train = torch.LongTensor(Y_train)
				Y_test = torch.LongTensor(Y_test)

				torch.save(X_train.clone().detach(), task_root+'X_train.pt', _use_new_zipfile_serialization=zipfile_save)
				torch.save(Y_train.clone().detach(), task_root+'Y_train.pt', _use_new_zipfile_serialization=zipfile_save)
				torch.save(X_test.clone().detach(), task_root+'X_test.pt', _use_new_zipfile_serialization=zipfile_save)
				torch.save(Y_test.clone().detach(), task_root+'Y_test.pt', _use_new_zipfile_serialization=zipfile_save)

			X_train, Y_train, X_test, Y_test = None, None, None, None
			del X_train
			del Y_train
			del X_test
			del Y_test
			
		elif regime == 'Open':
			assert general_preprocessing['n_open'], print('The number of IDs to hold out must be specified in the config JSON.')

			X_closed, Y_closed = None, None
			del X_closed
			del Y_closed

			if separator_preprocessing['mixing_to_memory'] is not None:
				
				mixing_params = separator_preprocessing['mixing_to_memory']

				X_test, Y_test, Y_test_id = generate_mixture_data(X_test, Y_test,
																  n_src=mixing_params['n_src'],
																  n_samples=mixing_params['validation_size'],
																  shift_factor=mixing_params['shift_factor'],
																  shift_overlaps=mixing_params['shift_overlaps'],
																  padding_scheme=mixing_params['padding_scheme'],
																  padding_side=mixing_params['padding_side'])
                
				task_directory = f'SeparatorOpen{n_src}Speakers/'

				task_root = root + task_directory
				if not os.path.isdir(task_root):
					os.mkdir(task_root)

				torch.save(X_test, task_root + 'X_test.pt')
				torch.save(Y_test, task_root + 'Y_test.pt')
				torch.save(Y_test_id, task_root + 'Y_test_id.pt')

				X_test, Y_test, Y_test_id = None, None, None
				del X_test
				del Y_test
				del Y_test_id
			else:

				task_directory = f'SeparatorOpen/'

				task_root = root + task_directory
				if not os.path.isdir(task_root):
					os.mkdir(task_root)
                    
				X_test = torch.Tensor(X_open)
				Y_test = torch.LongTensor(Y_open)
				torch.save(X_test.clone().detach(), task_root+'X_test.pt', _use_new_zipfile_serialization=zipfile_save)
				torch.save(Y_test.clone().detach(), task_root+'Y_test.pt', _use_new_zipfile_serialization=zipfile_save)

				X_test, Y_test, Y_test_id = None, None, None
				del X_test
				del Y_test
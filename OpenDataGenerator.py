import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys
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

def open_closed_split(X, Y, n_open, seed=42, save=None):
	random.seed(seed)
	
	ids = np.unique(Y).tolist()
	open_ids = random.sample(list(np.unique(Y)), n_open)
	closed_ids = [id for id in ids if id not in open_ids]
	
	mask = ~np.isin(Y, closed_ids)
	
	X_closed = np.delete(X, mask, axis=0)
	Y_closed = np.delete(Y, mask)

	X_open = X[mask]
	Y_open = Y[mask]
	
	if save is not None:
		if not os.path.isdir(save):
			os.mkdir(save)
		if save[-1] != r'/':
			save += r'/'
			
		save_split = 'OpenClosedSplit/'
		if not os.path.isdir(save + save_split):
			os.mkdir(save + save_split)
		
		np.save(save+save_split+'X_closed.npy', X_closed)
		np.save(save+save_split+'Y_closed.npy', Y_closed)
		
		np.save(save+save_split+'X_open.npy', X_open)
		np.save(save+save_split+'Y_open.npy', Y_open)
		
	X, Y = None, None
	del X
	del Y
		
	return X_closed, Y_closed, X_open, Y_open

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
									 side=kwargs['padding_side'])

		X_test, Y_test = augmenter(X_test, Y_test,
								   augmentation_factor=aug_factor,
								   shift_factor=kwargs['shift_factor'],
								   pad=kwargs['padding_scheme'],
								   side=kwargs['padding_side'])

	X_train = torch.Tensor(X_train)
	X_test = torch.Tensor(X_test)
	Y_train = torch.LongTensor(Y_train)
	Y_test = torch.LongTensor(Y_test)

	return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--os', type=str,
						help='specify Windows or Ubuntu', 
						default='Ubuntu')
	parser.add_argument('--data_directory', type=str,
						help='root directory for data files, e.g. Data')
	parser.add_argument('--animal', type=str,
						help='specify the animal to analyze')
	parser.add_argument('--objective', type=str,
						help='objective task, e.g. Classification, Separation, or Pipeline')
	parser.add_argument('--regime', type=str,
						help='open or closed regime, e.g. Open or Closed')
	parser.add_argument('--n_open', type=int,
						help='number of IDs to hold out from the separator')
	parser.add_argument('--n_src', type=int,
						help='number of sources',
						default=2)
	parser.add_argument('--n_individuals', type=int,
						help='number of individuals to include',
						default=None)
	parser.add_argument('--balance', type=str_to_bool,
						help='balance the datasets',
						default=False)
	parser.add_argument('--shift_factor', type=float,
						help='shift frame start fraction',
						default=None)
	parser.add_argument('--augmentation_factor', type=int,
						help='augmentation factor',
						default=None)
	parser.add_argument('--padding_scheme',
						help='padding scheme, e.g. zero, noise, or float',
						default=None)
	parser.add_argument('--padding_side',
						help='padding scheme side, e.g. front or back',
						default=None)
	parser.add_argument('--seed', type=int,
						default = 42)

	args = parser.parse_args()
	
	animal = args.animal
	assert animal in ['Macaque', 'Dolphin', 'Bat', 'SpermWhale']
	if not os.path.isdir(animal):
		os.mkdir(animal)
		
	root = args.data_directory
	if root[-1] != r'/':
		root += r'/'
	root = animal + '/' + root
	if not os.path.isdir(root):
		os.mkdir(root)
	
	random.seed(args.seed)
	np.random.seed(args.seed)
	
	task = args.objective
	assert task in ['Classification', 'Separation', 'Pipeline']

	regime = args.regime
	assert regime in ['Open', 'Closed']
	
	
	if not os.path.isdir(root+'OpenClosedSplit'):
		X, Y = generate_labeled_waveforms(animal,
										  os=args.os,
										  balance=args.balance,
										  n_individuals=args.n_individuals)
		X_closed, Y_closed, X_open, Y_open = open_closed_split(X, Y,
															   n_open=args.n_open,
															   seed=args.seed,
															   save=root)
	else:
		X_closed = np.load(root+'OpenClosedSplit/X_closed.npy')
		Y_closed = np.load(root+'OpenClosedSplit/Y_closed.npy')
		
		X_open = np.load(root+'OpenClosedSplit/X_open.npy')
		Y_open = np.load(root+'OpenClosedSplit/Y_open.npy')
		
	if task == 'Classification':
		X_open, Y_open = None, None
		del X_open
		del Y_open
		
		X_train, X_test, Y_train, Y_test = train_val_split(X_closed, Y_closed, seed=0,
														   augmentation_factor=args.augmentation_factor,
														   shift_factor=args.shift_factor,
														   padding_scheme=args.padding_scheme,
														   padding_side=args.padding_side)
		
		classifier_root = root + 'Classifier/'
		if not os.path.isdir(classifier_root):
			os.mkdir(classifier_root)

		np.save(classifier_root+'X_train.npy', X_train)
		np.save(classifier_root+'Y_train.npy', Y_train)
		np.save(classifier_root+'X_test.npy', X_test)
		np.save(classifier_root+'Y_test.npy', Y_test)
		
	else:
		if regime == 'Closed':
			X_open, Y_open = None, None
			del X_open
			del Y_open

			X_train, X_test, Y_train, Y_test = train_val_split(X_closed, Y_closed, seed=args.seed,
															   augmentation_factor=args.augmentation_factor,
															   shift_factor=args.shift_factor,
															   padding_scheme=args.padding_scheme,
															   padding_side=args.padding_side)
		
			if task == 'Separation':
				task_directory = 'SeparatorClosed/'
			elif task == 'Pipeline':
				task_directory = 'PipelineClosed/'
			
			task_root = root + task_directory
			if not os.path.isdir(task_root):
				os.mkdir(task_root)

			np.save(task_root+'X_train.npy', X_train)
			np.save(task_root+'Y_train.npy', Y_train)
			np.save(task_root+'X_test.npy', X_test)
			np.save(task_root+'Y_test.npy', Y_test)
			
		elif regime == 'Open':
			X_closed, Y_closed = None, None
			del X_closed
			del Y_closed
			
			X_test = torch.Tensor(X_open)
			Y_test = torch.LongTensor(Y_open)
			
			if task == 'Separation':
				task_directory = 'SeparatorOpen/'
			elif task == 'Pipeline':
				task_directory = 'PipelineOpen/'
			
			task_root = root + task_directory
			if not os.path.isdir(task_root):
				os.mkdir(task_root)

			np.save(task_root+'X_test.npy', X_test)
			np.save(task_root+'Y_test.npy', Y_test)
        

   
        
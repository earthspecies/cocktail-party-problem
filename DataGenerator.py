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

def generate_separator_data(X, Y, *args, **kwargs):
	mixer = SourceMixer(n_src=kwargs['n_src'], 
						samples=kwargs['n_samples'], 
						frames=X.shape[-1])

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

	train_subset, test_subset = mixer.train_test_subset(x_mix, 
														y_mix, 
														y_mix_id, 
														permute=kwargs['permute'])

	x_mix = None
	y_mix = None
	y_mix_id = None
	del x_mix
	del y_mix
	del y_mix_id

	return train_subset, test_subset

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
	parser.add_argument('--n_src', type=int,
						help='number of sources',
						default=2)
	parser.add_argument('--n_individuals', type=int,
						help='number of individuals to include',
						default=None)
	parser.add_argument('--balance', type=str_to_bool,
						help='balance the datasets',
						default=False)
	parser.add_argument('--n_samples', type=int,
						help='number of samples',
						default=10000)
	parser.add_argument('--shift_factor', type=float,
						help='shift frame start fraction',
						default=0.15)
	parser.add_argument('--augmentation_factor', type=int,
						help='augmentation factor',
						default=None)
	parser.add_argument('--shift_overlaps', type=str_to_bool,
						help='shift the overlapping calls',
						default=True)
	parser.add_argument('--padding_scheme',
						help='padding scheme, e.g. zero, noise, or float',
						default='zero')
	parser.add_argument('--padding_side',
						help='padding scheme side, e.g. front or back',
						default='front')
	parser.add_argument('--permute', type=str_to_bool,
						help='permute the data',
						default=True)
	parser.add_argument('--seed', type=int,
						default = 42)

	args = parser.parse_args()
    
	random.seed(args.seed)
	np.random.seed(args.seed)

	animal = args.animal
	assert animal in ['Macaque', 'Dolphin', 'Bat', 'SpermWhale', 'Elephant']
	root = args.data_directory
	if root[-1] != r'/':
		root += r'/'

	if animal == 'Macaque':
		loader = LoadMacaqueData(os=args.os)
		X, Y = loader.run(balance=args.balance)
	elif animal == 'Dolphin':
		loader = LoadDolphinData(os=args.os, n_individuals=args.n_individuals)
		X, Y = loader.run()
	elif animal == 'Bat':
		loader = LoadBatData(os=args.os)
		X, Y = loader.run(balance=args.balance)
	elif animal == 'SpermWhale':
		loader = LoadSpermWhaleData(os=args.os)
		X, Y = loader.run(balance=args.balance)
	elif animal == 'Elephant':
		loader = LoadElephantData(os=args.os)
		X, Y = loader.run()

	if not os.path.isdir(animal):
		os.mkdir(animal)

	root = animal + '/' + root
	if not os.path.isdir(root):
		os.mkdir(root)

	task = args.objective
	assert task in ['Classification', 'Separation', 'Pipeline']

	if task == 'Classification':
		X_train, X_test, Y_train, Y_test = generate_classifier_data(X, Y,
																	augmentation_factor=args.augmentation_factor,
																	shift_factor=args.shift_factor,
																	padding_scheme=args.padding_scheme,
																	padding_side=args.padding_side)

		classifier_root = root + 'Classifier/'
		if not os.path.isdir(classifier_root):
			os.mkdir(classifier_root)

		torch.save(X_train, classifier_root+'X_train.pt')
		torch.save(Y_train, classifier_root+'Y_train.pt')
		torch.save(X_test, classifier_root+'X_test.pt')
		torch.save(Y_test, classifier_root+'Y_test.pt')

	else:
		train_subset, test_subset = generate_separator_data(X, Y,
															n_src=args.n_src,
															n_samples=args.n_samples,
															shift_factor=args.shift_factor,
															shift_overlaps=args.shift_overlaps,
															padding_scheme=args.padding_scheme,
															padding_side=args.padding_side,
															permute=args.permute)

		X_train, Y_train, Y_train_id = train_subset
		X_test, Y_test, Y_test_id = test_subset

		if task == 'Separation':
			task_directory = 'Separator/'
		elif task == 'Pipeline':
			task_directory = 'Pipeline/'
            
		task_root = root + task_directory
		if not os.path.isdir(task_root):
			os.mkdir(task_root)

		torch.save(X_train, task_root+'X_train.pt')
		torch.save(Y_train, task_root+'Y_train.pt')
		torch.save(Y_train_id, task_root+'Y_train_id.pt')
		torch.save(X_test, task_root+'X_test.pt')
		torch.save(Y_test, task_root+'Y_test.pt')
		torch.save(Y_test_id, task_root+'Y_test_id.pt')

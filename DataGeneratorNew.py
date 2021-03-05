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

def open_closed_split(X, Y, n_open, seed=42):
	random.seed(seed)

	ids = np.unique(Y).tolist()
	open_ids = random.sample(list(np.unique(Y)), n_open)
	closed_ids = [id for id in ids if id not in open_ids]
	
	X_closed = [x for x,y in zip(X, Y) if y in closed_ids]
	Y_closed = [y for y in Y if y in closed_ids]
	
	X_open = [x for x,y in zip(X, Y) if y in open_ids]
	Y_open = [y for y in Y if y in open_ids]

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
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from itertools import permutations, product

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.xavier_uniform_(m.weight)

def n_src_channel_unroll(func):
	def unroll(tensor, *args, **kwargs):
		try:
			assert tensor.ndim==3
			result = [func(tensor[:, i, :].unsqueeze(dim=1), *args, **kwargs).unsqueeze(dim=1) for i in range(tensor.size(1))]
			return torch.cat(result, dim=1)
		except:
			tensor = tensor.unsqueeze(dim=0)
			result = [func(tensor[:, i, :].unsqueeze(dim=1), *args, **kwargs).unsqueeze(dim=1) for i in range(tensor.size(1))]
			return torch.cat(result, dim=1).squeeze(dim=0)
	return unroll

@n_src_channel_unroll
def stft_transform(x, stft):
	_, mag = stft(x)
	return mag

def pit_wrapper_loss(loss_fn):
	def permutation_loss(y_pred, y_true, *args, **kwargs):
		try:
			batch, channels, samples = y_true.size()
		except:
			batch, channels, height, width = y_true.size()

		perms = list(permutations(tuple(range(channels))))
		perm_loss = sum([min([loss_fn(y_pred[i, p].unsqueeze(dim=0), y_true[i].unsqueeze(dim=0), *args, **kwargs) for p in perms]) for i in range(batch)])/batch
		return perm_loss
	return permutation_loss

def pit_wrapper_metric(metric_fn):
	def permutation_metric(y_pred, data, index, *args, **kwargs):
		dims = data[index].ndim
		size = data[index].size()
		if dims == 2:
			batch, channels = size
		elif dims == 3:
			batch, channels, samples = size
		elif dims == 4:
			batch, channels, height, width = size

		perms = list(permutations(tuple(range(channels))))
		perm_metric = sum([max([metric_fn(y_pred[i, p], data[index], i, *args, **kwargs) for p in perms]) for i in range(batch)])/batch
		return perm_metric
	return permutation_metric

def augmenter(X, Y, augmentation_factor, shift_factor, pad='zero', side='front', shuffle=True):
	assert type(augmentation_factor)==int, print('Augmentation factor must be an integer')
	if type(pad)==str:
		assert pad in ['zero', 'noise', 'sub_noise'], print('Pad must be \'zero\', \'noise\', \'sub_noise\' or a float')
	else:
		assert type(pad) == float, print('Pad must be \'zero\', \'noise\', \'sub_noise\' or a float')
	assert side in ['front', 'back']
    
	frames = X[0].shape[-1]
	samples = len(X)
    
	X_aug, Y_aug = [], []
    
	for i in range(samples):
		x_i = X[i]
		y_i = Y[i]
		if pad == 'zero':
			noise = 0
		elif pad == 'noise':
			noise = X[i, -1]
		elif pad == 'sub_noise':
			noise = X[i, -1]
			x_i = x_i - noise
			noise = 0
		else:
			noise = pad
		for j in range(augmentation_factor):
			frame_shift = random.randint(0, int(shift_factor * frames))
			noise_j = np.ones(frame_shift, dtype='float32') * noise
            
			if side == 'front':
				x_j = x_i[0:frames - frame_shift]
			elif side == 'back':
				x_j = x_j[frame_shift-1:-1]
            
			x_j = np.concatenate([noise_j, x_j]).astype('float32')
            
			X_aug.append(x_j)
			Y_aug.append(y_i)
	X, Y = None, None
	del X
	del Y
	if shuffle:
		data = list(zip(X_aug, Y_aug))
		random.shuffle(data)
		X_aug, Y_aug = zip(*data) 
	return X_aug, Y_aug

def nll_loss_weights(Y):
	labels, counts = np.unique(Y, return_counts=True)
	weights = np.max(counts) / counts
	weights /= np.max(weights)
	return weights

def id_mapper(y):
	try:
		ids = np.unique(y.numpy()).tolist()
		id_dict = {e:i for i,e in enumerate(ids)}
		for i in range(y.size(0)):
			y[i] = id_dict[y[i].item()]
	except AttributeError:
		ids = np.unique(y).tolist()
		id_dict = {e:i for i,e in enumerate(ids)}
		for i in range(len(y)):
			y[i] = id_dict[y[i]]
	return y
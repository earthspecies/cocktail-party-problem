import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from Utils import *
from Layers import *

class PipelineDataset(torch.utils.data.Dataset):
	def __init__(self, X, Y, Y_id, n_src=2, objective='classification', stft_params=None, filter_params=None, shuffle=False):
		super(PipelineDataset, self).__init__()
		self.X = X
		self.Y = Y
		self.Y_id = Y_id
		self.objective = objective
		self.n_src = n_src
		self.shuffle = shuffle
		
		if objective == 'classification':
			self.len = n_src*X.size(0)
			self.X_classification = Y.view(-1, Y.size(-1)).unsqueeze(dim=1)
			self.Y_classification = Y_id.view(-1)
			if self.shuffle:
				torch.manual_seed(42)
				p = torch.randperm(self.X_classification.size(0))
				self.X_classification = self.X_classification[p, :, :]
				self.Y_classification = self.Y_classification[p]
		elif self.objective == 'separation':
			self.len = X.size(0)

		if filter_params is not None:
			self.filter = HighPassFilter(cutoff_freq=filter_params['cutoff_freq'],
										 sample_rate=filter_params['sample_rate'],
										 b=filter_params['b'])
		else:
			self.filter = lambda x: x
			
		if stft_params is not None:
			stft = STFT(stft_params['kernel_size'],
						stft_params['stride'],
						stft_params['dB'])
			self.representation = lambda x: stft_transform(x, stft)
		else:
			self.representation = lambda x: x
		
		self.transforms = lambda z: reduce((lambda x, fx: fx(x)), [n_src_channel_unroll(self.filter), torch.squeeze, self.representation], z)

	def __len__(self):
		return self.len
	
	def __getitem__(self, idx):
		if self.objective == 'classification':
			return self.X_classification[idx], self.Y_classification[idx]
		
		elif self.objective == 'separation':
			return self.X[idx], self.transforms(self.Y[idx]), self.Y_id[idx]


class ClassifierDataset(torch.utils.data.Dataset):
	def __init__(self, X, Y):
		super(ClassifierDataset, self).__init__()
		self.X = X
		self.Y = Y

		self.len = X.size(0)

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx]


class SeparatorDataset(torch.utils.data.Dataset):
	def __init__(self, X, Y, Y_id, stft_params=None, filter_params=None):
		super(SeparatorDataset, self).__init__()
		self.X = X
		self.Y = Y
		self.Y_id = Y_id
		
		self.len = X.size(0)

		if filter_params is not None:
			self.filter = HighPassFilter(cutoff_freq=filter_params['cutoff_freq'],
										 sample_rate=filter_params['sample_rate'],
										 b=filter_params['b'])
		else:
			self.filter = lambda x: x
			
		if stft_params is not None:
			stft = STFT(stft_params['kernel_size'],
						stft_params['stride'],
						stft_params['dB'])
			self.representation = lambda x: stft_transform(x, stft)
		else:
			self.representation = lambda x: x
		
		self.transforms = lambda z: reduce((lambda x, fx: fx(x)), [n_src_channel_unroll(self.filter), torch.squeeze, self.representation], z)

	def __len__(self):
		return self.len
	
	def __getitem__(self, idx):
		return self.X[idx], self.transforms(self.Y[idx]), self.Y_id[idx]
    
class MixtureDataset(torch.utils.data.Dataset):
	def __init__(self, X, Y, size=None, n_src=2, subset='train', seed=42):
		self.Y = Y
		self.categories = torch.unique(Y).tolist()
		self.X_cat = {c:X[self.Y == c] for c in self.categories}

		if size is None:
			self.size = X.size(0) // n_src
		else:
			self.size = size
            
		self.n_src = n_src
		self.subset = subset
		self.seed = seed
		random.seed(self.seed)

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		if self.subset == 'val':
			random.seed(idx)

		ids = random.sample(self.categories, self.n_src)

		Xs = [random.choice(self.X_cat[i]) for i in ids]

		Ys = [torch.LongTensor([i]) for i in ids]

		X_mix = sum(Xs).unsqueeze(dim=0)
		Y_mix = torch.vstack(Xs)
		Y_mix_id = torch.vstack(Ys).squeeze(dim=-1)

		return X_mix, Y_mix, Y_mix_id
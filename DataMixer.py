import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.model_selection import train_test_split

class SourceMixer(object):
	def __init__(self, n_src, samples, frames, seed=42):
		self.n_src = n_src
		self.samples = samples
		self.frames = frames
		self.seed = seed
		
	def mix(self, x, y, shift_factor=None, shift_overlaps=False, **kwargs):
		random.seed(self.seed)
		np.random.seed(self.seed)
		
		ids, counts = np.unique(y, return_counts=True)
		ids, counts = ids.tolist(), counts.tolist()
		
		x_mix = np.zeros((self.samples, 1, self.frames), dtype='float32')
		y_mix = np.zeros((self.samples, self.n_src, self.frames), dtype='float32')
		y_mix_id = np.zeros((self.samples, self.n_src), dtype='int64')
		
		for n in tqdm.tqdm(range(self.samples)):
			id_idxs = random.sample(ids, self.n_src)
			signal_idxs = [np.random.randint(low=0, 
											 high=counts[ids.index(id_idx)]) for id_idx in id_idxs]
			
			signals = [x[y==id_idx][signal_idx] for (id_idx, signal_idx) in zip(id_idxs, signal_idxs)]
			
			if shift_factor is not None:
				signals = [self.signal_shifter(signal,
											   shift_factor=shift_factor,
											   pad=kwargs['pad'],
											   side=kwargs['side']) for signal in signals]
				
			if shift_overlaps:
				assert kwargs['pad'] == 'zero', print('Overlap shift is compatible with zero pad only')

				signals = self.overlap_shifter(signals)

			for j in range(self.n_src):
				y_mix[n, j, :] = signals[j]
				y_mix_id[n, j] = id_idxs[j]

			x_mix[n, 0, :] = sum(signals)

		return x_mix, y_mix, y_mix_id
	
	def train_test_subset(self, x_mix, y_mix, y_mix_id, split=0.2, permute=False):
		np.random.seed(self.seed)
		random.seed(self.seed)

		y_concat = np.zeros((y_mix.shape[0], y_mix.shape[1], y_mix.shape[-1] + 1), dtype='float32')
		y_concat = np.concatenate([y_mix, np.expand_dims(y_mix_id, axis=-1)], axis=-1, out=y_concat)

		x_train, x_test, y_train, y_test = train_test_split(x_mix, y_concat, test_size=split, random_state=self.seed)

		if permute:
			perms = list(permutations(np.arange(self.n_src)))
			n_perms = len(perms)
			for i in range(y_train.shape[0]):
				k = random.randint(0, n_perms-1)
				y_train[i, :, :] = y_train[i, perms[k], :]

			for i in range(y_test.shape[0]):
				k = random.randint(0, n_perms-1)
				y_test[i, :, :] = y_test[i, perms[k], :]

		y_train_id = y_train[:, :, -1]
		y_test_id = y_test[:, :, -1]

		y_train = y_train[:, :, :-1]
		y_test = y_test[:, :, :-1]

		train_subset = [torch.Tensor(x_train), torch.Tensor(y_train), torch.LongTensor(y_train_id)]
		test_subset = [torch.Tensor(x_test), torch.Tensor(y_test), torch.LongTensor(y_test_id)]

		return train_subset, test_subset
	
	@staticmethod
	def signal_shifter(signal, shift_factor, pad='zero', side='front'):
		if type(pad)==str:
			assert pad in ['zero', 'noise', 'sub_noise'], print('Pad must be \'zero\', \'noise\', \'sub_noise\' or a float')
		else:
			assert type(pad) == float, print('Pad must be \'zero\', \'noise\', \'sub_noise\' or a float')
		assert side in ['front', 'back']

		frames = signal.shape[-1]

		if pad == 'zero':
			noise = 0
		elif pad == 'noise':
			noise = signal[-1]
		elif pad == 'sub_noise':
			noise = signal
			signal = signal - noise
			noise = 0
		else:
			noise = pad
			
		frame_shift = random.randint(0, int(shift_factor * frames))
		noise_frames = np.ones(frame_shift, dtype='float32') * noise

		if side == 'front':
			signal_frames = signal[0:frames - frame_shift]
		elif side == 'back':
			signal_frames = signal[frame_shift-1:-1]

		signal = np.concatenate([noise_frames, signal_frames]).astype('float32')

		return signal
	
	@staticmethod
	def overlap_shifter(signals):
		frames = signals[0].shape[-1]

		back_trimmed = [np.trim_zeros(s, 'b') for s in signals]

		back_trimmed_lengths = [len(b) for b in back_trimmed]
		max_length = max(back_trimmed_lengths)
		max_index = back_trimmed_lengths.index(max_length)

		front_zeros = [frames - len(np.trim_zeros(s, 'f')) for s in signals]

		for i, s in enumerate(signals):
			if i != max_index:
				max_shift = max_length - front_zeros[i]
				min_shift = max(0, front_zeros[max_index] - back_trimmed_lengths[i])
				shift = np.zeros(np.random.randint(low=min_shift, high=np.max([max_shift, min_shift+1])), dtype='float32')
				signals[i] = np.concatenate([shift, s]).astype('float32')[:frames]
		return signals
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tqdm

class HighPassFilter(nn.Module):
	def __init__(self, cutoff_freq, sample_rate, b=0.08, eps=1e-20):
		super(HighPassFilter, self).__init__()
		self.fc = cutoff_freq / sample_rate
		self.b = b
		
		N = int(np.ceil((4 / b)))
		if not N % 2:
			N+=1
		self.N = N
		
		self.epsilon = nn.Parameter(torch.tensor(eps), requires_grad=False)
		self.window = nn.Parameter(torch.blackman_window(N), requires_grad=False)
		
		n = torch.arange(N)
		self.sinc_fx = nn.Parameter(self.sinc(2 * self.fc * (n - (self.N-1) / 2.)), requires_grad=False)
		
	def forward(self, x):
		x = x.view(x.size(0), 1, x.size(-1))
		sinc_fx = self.sinc_fx * self.window
		sinc_fx = torch.true_divide(sinc_fx, torch.sum(sinc_fx))
		sinc_fx = -sinc_fx
		sinc_fx[int((self.N - 1) / 2)] += 1
		output = torch.nn.functional.conv1d(x, sinc_fx.view(-1, 1, self.N), padding=self.N//2)
		return output
		
	def sinc(self, x):
		y = np.pi*torch.where(x==0, self.epsilon, x)
		return torch.true_divide(torch.sin(y), y)  

class STFT(nn.Module):
	def __init__(self, kernel_size, stride, dB=False, epsilon=1e-8):
		super(STFT, self).__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.window = nn.Parameter(torch.hann_window(kernel_size), requires_grad=False)
		self.epsilon = epsilon
		self.dB = dB

	def forward(self, x):
		S = torch.stft(x.squeeze(dim=1), 
					   n_fft=self.kernel_size, 
					   hop_length=self.stride, 
					   window=self.window)
		S_real = S[:, :, :, 0] + self.epsilon
		S_imag = S[:, :, :, 1] + self.epsilon
		P = torch.atan2(S_imag, S_real)
		D = torch.sqrt(torch.add(torch.pow(S_real, 2), torch.pow(S_imag, 2)))
		if self.dB:
			D = self.amplitude_to_db(D)
		return P, D

	def get_out_size(self, in_size):
		batch, in_filters, L_in = in_size
		L_out = L_in // self.stride + 1
		out_filters = self.kernel_size // 2 + 1
		return (batch, out_filters, L_out)

	def get_config(self):
		config = {
			'name': 'STFT',
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dB scaling': self.dB
		}
		return config

	@staticmethod
	def amplitude_to_db(S, amin=1e-10):
		S = S + amin
		D = torch.mul(torch.log10(S), 20)
		return D

class iSTFT(nn.Module):
	def __init__(self, kernel_size, stride, dB=False):
		super(iSTFT, self).__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.window = nn.Parameter(torch.hann_window(kernel_size), requires_grad=False)
		self.dB = dB

	def forward(self, P, D):
		if self.dB:
			D = self.db_to_amplitude(D)
		S_real = torch.mul(D, torch.cos(P)).unsqueeze(dim=-1)
		S_imag = torch.mul(D, torch.sin(P)).unsqueeze(dim=-1)
		S = torch.cat([S_real, S_imag], dim=-1)

		x = torch.istft(S, n_fft=self.kernel_size, hop_length=self.stride, window=self.window).unsqueeze(dim=1)
		return x

	def get_out_size(self, in_size):
		batch, in_filters, L_in = in_size
		L_out = int(L_in - 1) * self.stride
		return (batch, 1, L_out)

	def get_config(self):
		config = {
			'name': 'iSTFT',
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dB scaling': self.dB
		}

	@staticmethod
	def db_to_amplitude(D, amin=1e-10):
		S = torch.pow(10, torch.true_divide(D, 20)) - amin
		return S

class Padding2D(nn.Module):
	def __init__(self, in_size, x_factor=1, y_factor=1):
		super(Padding2D, self).__init__()
		self.x_factor = x_factor
		self.y_factor = y_factor
		
		self.x_pad = self.add_padding(in_size[-1], x_factor)
		self.y_pad = self.add_padding(in_size[-2], y_factor)
			
		
	def forward(self, x):
		ydim, xdim = x.size()[-2:]

		x = F.pad(x, (0, self.x_pad, 0, self.y_pad, 0, 0))
		return x
	
	@staticmethod
	def add_padding(size, factor):
		pad = int(np.ceil(size / factor) * factor) - size
		return pad

class Padding1D(nn.Module):
	def __init__(self, pad):
		super(Padding1D, self).__init__()
		self.pad = pad
		
	def forward(self, x):
		x = F.pad(x, (0, self.pad))
		return x

class Cropping2D(nn.Module):
	def __init__(self, x_crop, y_crop):
		super(Cropping2D, self).__init__()
		self.x_crop = x_crop
		self.y_crop = y_crop
	
	def forward(self, x):
		x = torch.split(x, [x.size(-2) - self.y_crop, self.y_crop], dim=-2)[0]
		x = torch.split(x, [x.size(-1) - self.x_crop, self.x_crop], dim=-1)[0]
		return x

class Cropping1D(nn.Module):
	def __init__(self, crop):
		super(Cropping1D, self).__init__()
		self.crop = crop
	
	def forward(self, x):
		x = torch.split(x, [x.size(-1) - self.crop, self.crop], dim=-1)[0]
		return x

class SplitChannels(nn.Module):
	def __init__(self, n_src=2):
		super(SplitChannels, self).__init__()
		self.n_src = n_src
		
	def forward(self, x):
		split_size = [1 for _ in range(self.n_src)]
		x = torch.split(x, split_size_or_sections=split_size, dim=1)
		return x

class MCNN(nn.Module):
	def __init__(self, n_heads=8, **kwargs):
		super(MCNN, self).__init__()
		self.n_heads = n_heads
		self.heads = nn.ModuleList([self.construct_head(layers=kwargs['layers'],
														in_filters=kwargs['in_filters'],
														K=kwargs['K'],
														s=kwargs['s'],
														D=kwargs['D']) for _ in range(n_heads)])
		self.ws = nn.Parameter(torch.ones(n_heads), requires_grad=True)
		self.a = nn.Parameter(torch.ones(1), requires_grad=True)
		self.b = nn.Parameter(torch.ones(1), requires_grad=True)
	
	def forward(self, x):
		x = sum([wi*hi(x) for wi, hi in zip(self.ws, self.heads)])
		x = self.LearnableSoftsign(self.a, self.b, x)
		return x
	
	@staticmethod
	def LearnableSoftsign(a, b, x):
		out = a * x / (1 + torch.abs(b * x))
		return out
	
	@staticmethod
	def construct_head(layers, in_filters, K, s, D):
		P = int(D * (K - 1) / 2)
		c_ins = [2**(layers-i) for i in range(1, layers)]
		c_ins = [in_filters, *c_ins]
		c_outs = [2**(layers-i) for i in range(1, layers+1)]
		layers = []
		for c_i, c_o in zip(c_ins, c_outs):
			layers.append(nn.ConvTranspose1d(c_i, c_o, K, s, padding=P))
			layers.append(nn.ELU())
		layers.append(Cropping1D(1))
		head = nn.Sequential(*layers)
		return head


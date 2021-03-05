import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import re

from Layers import *

class Classifier(nn.Module):
	def __init__(self, 
				 in_size, 
				 n_classes, 
				 n_blocks, 
				 pool_size=2, 
				 input_mode='raw',
				 lin_dim=128,
				 dropout=0.5,
				 first_filter=32,
				 filters=64,
				 stft_params={'kernel_size':1024, 'stride':1024//16}, 
				 **kwargs):
		super(Classifier, self).__init__()
		self.n_classes = n_classes
		self.input_mode = input_mode
		self.n_blocks = n_blocks
		self.pool_size = pool_size
		self.lin_dim = lin_dim
		self.dropout = dropout
		self.first_filter = first_filter
		self.filters = filters

		nfft = stft_params['kernel_size']
		hop = stft_params['stride']
		
		try:
			filter_params = kwargs['filter_params']
			self.filter = HighPassFilter(**filter_params)
		except KeyError:
			self.filter = lambda x: x

		if input_mode == 'raw':
			self.transform = STFT(nfft, hop, dB=False)
			out_size = self.transform.get_out_size(in_size)[-2:]
			self.lin_size = self.get_lin_size(shape=out_size, 
											  blocks=n_blocks, 
											  pool=pool_size,
											  filters=filters)
		elif input_mode == 'raw_db':
			self.transform = STFT(nfft, hop, dB=True)
			out_size = self.transform.get_out_size(in_size)[-2:]
			self.lin_size = self.get_lin_size(shape=out_size, 
											  blocks=n_blocks, 
											  pool=pool_size,
											  filters=filters)
		elif input_mode == 'stft':
			self.transform = STFT(nfft, hop, dB=False).amplitude_to_db
			out_size = in_size[-2:]
			self.lin_size = self.get_lin_size(shape=out_size, 
											  blocks=n_blocks, 
											  pool=pool_size,
											  filters=filters)
		elif input_mode == 'stft_db':
			self.transform = lambda x: x
			out_size = in_size[-2:]
			self.lin_size = self.get_lin_size(shape=out_size, 
											  blocks=n_blocks, 
											  pool=pool_size,
											  filters=filters)

		block_list = [self.cnn_conv_block(1, first_filter, pool=pool_size),
					  self.cnn_conv_block(first_filter, filters, pool=pool_size),
					  nn.Dropout(dropout)]
		for b in range(n_blocks-2):
			block_list.append(self.cnn_conv_block(filters, filters, pool=pool_size))
			block_list.append(nn.Dropout(dropout))

		self.blocks = nn.ModuleList(block_list)

		self.lin1 = nn.Linear(self.lin_size, lin_dim)
		self.lin2 = nn.Linear(lin_dim, n_classes)
		self.drop = nn.Dropout(dropout)
	def forward(self, x):
		x = self.filter(x)
		if 'raw' in self.input_mode:
			_, x = self.transform(x)
			x = x.unsqueeze(dim=1)
		else:
			x = self.transform(x)
		for b in self.blocks:
			x = b(x)
		x = x.view(-1, self.lin_size)
		x = F.leaky_relu(self.lin1(x))
		x = self.drop(x)
		x = self.lin2(x)
		x = F.log_softmax(x, dim=1)
		return x
	
	@staticmethod
	def cnn_conv_block(in_filters, out_filters, pool):
		block = nn.Sequential(
			nn.Conv2d(in_channels=in_filters,
					  out_channels=out_filters,
					  kernel_size=(3,3),
					  padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(in_channels=out_filters,
					  out_channels=out_filters,
					  kernel_size=(3,3),
					  padding=1),
			nn.LeakyReLU(),
			nn.MaxPool2d(kernel_size=(pool, pool)))
		return block
	
	@staticmethod
	def get_lin_size(shape, blocks, pool, filters):
		for j in range(blocks):
			shape = [np.floor(s / pool) for s in shape]
		result = filters * shape[0] * shape[1]
		return int(result)

class UNet2D(nn.Module):
	def __init__(self, in_size, n_src, n_blocks, pool_size=2, batch_norm=False):
		super(UNet2D, self).__init__()
		self.n_src = n_src
		self.n_blocks = n_blocks
		self.in_size = in_size
		self.batch_norm = batch_norm
		
		self.padding2d = Padding2D(in_size=in_size,
								   x_factor=pool_size**n_blocks,
								   y_factor=pool_size**n_blocks)
	   
		self.downs = nn.ModuleList([self.conv_block(1, 2**(i+4), batch_norm=batch_norm) if i==0 
									else self.conv_block(2**(i+3), 2**(i+4), batch_norm=batch_norm) 
									for i in range(n_blocks)])
		self.maxpool = nn.MaxPool2d(kernel_size=(pool_size,pool_size))

		middle1 = [
			nn.Conv2d(in_channels=2**(4+(n_blocks-1)), 
					  out_channels=2**(4+(n_blocks-1)), 
					  kernel_size=(3,3),
					  padding=1),
			nn.LeakyReLU()]

		middle2 = [
			nn.Conv2d(in_channels=2**(4+(n_blocks-1)), 
					  out_channels=2**(4+(n_blocks-1)), 
					  kernel_size=(3,3),
					  padding=1),
			nn.LeakyReLU()]

		if batch_norm:
			middle1.append(nn.BatchNorm2d(2**(4+(n_blocks-1))))
			middle2.append(nn.BatchNorm2d(2**(4+(n_blocks-1))))
		
		
		self.conv_middle1 = nn.Sequential(*middle1)
		self.conv_middle2 = nn.Sequential(*middle2)
		
		self.ups = nn.ModuleList([self.conv_block(*filters, batch_norm=batch_norm)
								  for filters in self.compute_upfilters(n_blocks)])
		self.upsample = nn.Upsample(scale_factor=pool_size,
									mode='bilinear',
									align_corners=True)
		
		self.conv_last = nn.Conv2d(in_channels=16,
								   out_channels=n_src,
								   kernel_size=(1,1))

		self.cropping2d = Cropping2D(self.padding2d.x_pad,
									 self.padding2d.y_pad)
		self.splitchannels = SplitChannels(n_src=n_src)
		
	def forward(self, x):
		x = self.padding2d(x)
		down1 = self.downs[0](x)
		pool1 = self.maxpool(down1)
		xdowns, xpools = [down1], [pool1]
		for i in range(1, self.n_blocks):
			xdowns.append(self.downs[i](xpools[i-1]))
			xpools.append(self.maxpool(xdowns[-1]))
		xmiddle = self.conv_middle1(xpools[-1])
		xmiddle = self.conv_middle2(xmiddle)
		upsample_n_block = self.upsample(xmiddle)
		concat = torch.cat([upsample_n_block, xdowns[self.n_blocks-1]], dim=1)
		up_n_block = self.ups[0](concat)
		xupsamples, xups = [upsample_n_block], [up_n_block]
		for i in range(1, self.n_blocks):
			xupsamples.append(self.upsample(xups[-1]))
			concat = torch.cat([xupsamples[-1], xdowns[self.n_blocks - (i+1)]], dim=1)
			xups.append(self.ups[i](concat))
		x = self.conv_last(xups[-1])
		x = self.cropping2d(x)
		x = self.splitchannels(x)
		return x
		
	@staticmethod
	def conv_block(in_filters, out_filters, batch_norm=False):
		layers = [nn.Conv2d(in_channels=in_filters,
					  out_channels=out_filters,
					  kernel_size=(3,3),
					  padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(in_channels=out_filters,
					  out_channels=out_filters,
					  kernel_size=(3,3),
					  padding=1),
			nn.LeakyReLU()
		]
		if batch_norm:
			layers.insert(2, nn.BatchNorm2d(out_filters))
			layers.insert(5, nn.BatchNorm2d(out_filters))
		block = nn.Sequential(*layers)
		return block
	
	@staticmethod
	def compute_upfilters(n_blocks):
		in_filter = 2**(4 + (n_blocks-1))
		filters = [((in_filter + in_filter), in_filter)]
		for i in range(n_blocks - 1):
			in_filter = filters[i][-1]
			out_filter = 2**(4 + (n_blocks-1) - (i+1))
			filters.append((in_filter + out_filter, out_filter))
		return filters

class RepUNet(nn.Module):
	def __init__(self, in_size, n_src, filterbank_params={'nfft':1024, 'hop':1024//16}, input_mode='conv1d', output_mode='conv1d', phase_channel=False, **kwargs):
		super(RepUNet, self).__init__()
		
		nfft = filterbank_params['nfft']
		hop = filterbank_params['hop']
		
		self.n_src = n_src
		self.in_size = in_size
		self.input_mode = input_mode
		self.output_mode = output_mode
		
		try:
			filter_params = kwargs['filter_params']
			self.filter = HighPassFilter(**filter_params)
		except KeyError:
			self.filter = lambda x: x
			
		if input_mode=='conv1d':
			self.representation = nn.Conv1d(in_channels=1, 
											out_channels=nfft // 2 + 1, 
											kernel_size=nfft, 
											stride=hop,
											padding=nfft//2)
			try:
				dropout = kwargs['dropout']
				self.drop = nn.Dropout(dropout)
			except KeyError:
				self.drop = None

			self.zero_phase = lambda x: torch.nn.Parameter(torch.zeros(x.size(), device=x.device), requires_grad=False)
		elif input_mode=='stft':
			self.representation = STFT(nfft, hop, False)
		elif input_mode=='stft_db':
			self.representation = STFT(nfft, hop, True)
		
		start_size = (None, 1, nfft // 2 + 1, self.conv1d_out_size(in_size=in_size[-1],
												 k=nfft,
												 s=hop,
												 p=nfft//2))
		self.unet2d = UNet2D(in_size=start_size,
							 n_src=n_src,
							 n_blocks=kwargs['n_blocks'],
							 pool_size=kwargs['pool_size'],
							 batch_norm=kwargs['batch_norm'])
		self.phase_channel = phase_channel
		if self.phase_channel:
			assert re.search('istft', output_mode), 'Phase channel must be used in combination with the iSTFT'
			self.unet2dphase = UNet2D(in_size=start_size,
									  n_src=n_src,
									  n_blocks=kwargs['n_blocks'],
									  pool_size=kwargs['pool_size'],
									  batch_norm=kwargs['batch_norm'])
		
		if output_mode == 'conv1d':
			self.conv1d_ts = nn.ModuleList([nn.ConvTranspose1d(in_channels=nfft//2 + 1,
															   out_channels=1,
															   kernel_size=nfft,
															   stride=hop,
															   padding=nfft//2,
															   output_padding=0) for _ in range(n_src)])
		elif output_mode == 'istft':
			self.istft = iSTFT(nfft, hop, False)
		elif output_mode == 'istft_db':
			self.istft = iSTFT(nfft, hop, True)
		elif re.search('mcnn_\d+', output_mode):
			n_h = int(re.findall('\d+', output_mode)[0])
			self.mcnns = nn.ModuleList([MCNN(n_heads=n_h, **kwargs['mcnn_params']) for _ in range(n_src)])

		self.padding1d = Padding1D(pad=in_size[-1] - self.conv1dtranspose_out_size(self.conv1d_out_size(in_size=in_size[-1],
																										k=nfft,
																										s=hop,
																										p=nfft//2),
																				   k=nfft, 
																				   s=hop,
																				   pad=nfft//2))         
	def forward(self, x):
		x = self.filter(x)

		if self.input_mode == 'conv1d':
			filterbank = self.representation(x)
			if self.drop is not None:
				filterbank = self.drop(filterbank)
			phase = self.zero_phase(filterbank)
		else:
			phase, filterbank = self.representation(x)
	
		filterbank = torch.unsqueeze(filterbank, dim=1)
		
		if self.phase_channel:
			phase = torch.unsqueeze(phase, dim=1)
			phase = self.unet2dphase(phase)
			phase = [p.squeeze() for p in phase]
		else:
			phase = [phase for _ in range(self.n_src)]
		
		x = self.unet2d(filterbank)
		
		if self.output_mode == 'conv1d':
			x = [torch.mul(x_i, filterbank).squeeze(dim=1) for x_i in x]
			x = [conv1d_ti(x_i) for x_i, conv1d_ti in zip(x, self.conv1d_ts)]
			x = [self.padding1d(x_i) for x_i in x]
			x = torch.cat(x, dim=1)

		elif (self.output_mode == 'istft') or (self.output_mode == 'istft_db'):
			x = [torch.mul(F.relu(x_i), filterbank).squeeze(dim=1) for x_i in x]
			x = [self.istft(p_i, x_i) for p_i, x_i in zip(phase, x)]
			x = [self.padding1d(x_i) for x_i in x]
			x = torch.cat(x, dim=1)
		elif re.search('mcnn_\d+', self.output_mode):
			x = [torch.mul(x_i, filterbank).squeeze(dim=1) for x_i in x]
			x = [mcnn_i(x_i) for x_i, mcnn_i in zip(x, self.mcnns)]
			x = [self.padding1d(x_i) for x_i in x]
			x = torch.cat(x, dim=1)
		elif self.output_mode == 'spec':
			x = [torch.mul(F.relu(x_i), filterbank) for x_i in x]
			x = torch.cat(x, dim=1)
		return x
	
	@staticmethod
	def conv1d_out_size(in_size, k, s, p):
		return (in_size - (k-1) + 2*p) // s + 1

	@staticmethod
	def conv1dtranspose_out_size(in_size, k, s, pad, opad=0):
		out = (in_size-1)*s + k-2*pad + opad
		return out
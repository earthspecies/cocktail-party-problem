import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import *

def mae_loss(y_pred, y_true):
	"""L1 Loss"""
	e = torch.mean(torch.abs(y_true - y_pred))
	return e

def mse_loss(y_pred, y_true):
	"""L2 Loss"""
	e = torch.mean(torch.square(y_true - y_pred))
	return e

def raw2spec_mae_loss(y_pred, y_true, stft):
	"""L1 Loss on raw2spec Spectrograms"""
	stft = stft.to(y_pred.device)
	y_pred_2spec = stft_transform(y_pred, stft)
	y_true_2spec = stft_transform(y_true, stft)
	
	e = torch.mean(torch.abs(y_true_2spec - y_pred_2spec))
	return e

def raw2spec_mse_loss(y_pred, y_true, stft):
	"""Le Loss on raw2spec Spectrograms"""
	stft = stft.to(y_pred.device)
	y_pred_2spec = stft_transform(y_pred, stft)
	y_true_2spec = stft_transform(y_true, stft)
	
	e = torch.mean(torch.square(y_true_2spec - y_pred_2spec))
	return e

def fnorm(matrix):
	"""Frobenius Norm"""
	f = torch.sqrt(torch.sum(torch.pow(torch.abs(matrix), 2)))
	return f

def spectral_convergence_loss(y_pred, y_true):
	"""Spectral Convergence Loss on Spectrograms"""
	e = fnorm(y_true - y_pred) / fnorm(y_true)
	return e

def raw2spec_spectral_convergence_loss(y_pred, y_true, stft):
	"""Spectral Convergence Loss on raw2spec Spectrograms"""
	stft = stft.to(y_pred.device)
	assert stft.dB==False, print('Spectral convergence uses linearly scaled spectrograms')
	y_pred_2spec = stft_transform(y_pred, stft)
	y_true_2spec = stft_transform(y_true, stft)
	
	e = fnorm(y_true_2spec - y_pred_2spec) / fnorm(y_true_2spec)
	return e

def neg_si_sdr(y_pred, y_true, zero_mean=False, epsilon=1e-10):
	if zero_mean:
		mean_true = torch.mean(y_true, dim=2, keepdim=True)
		mean_pred = torch.mean(y_pred, dim=2, keepdim=True)
		y_true = y_true - mean_true
		y_pred = y_pred - mean_pred
	
	pairwise_dot = torch.sum(y_pred * y_true, dim=2, keepdim=True)
	true_energy = torch.sum(y_true ** 2, dim=2, keepdim=True) + epsilon
	scaled_true = pairwise_dot * y_true / true_energy
	
	e_noise = y_pred - scaled_true
	
	pairwise_sdr = torch.sum(scaled_true ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + epsilon)
	pairwise_sdr = 10 * torch.log10(pairwise_sdr + epsilon)
	return -torch.mean(pairwise_sdr)

def total_loss(y_pred, y_true, stft):
	stft = stft.to(y_pred.device)
	assert stft.dB==False, print('Spectral convergence uses linearly scaled spectrograms')
	y_pred_2spec = stft_transform(y_pred, stft)
	y_true_2spec = stft_transform(y_true, stft)

	w1 = 1
	w2 = 1e-1
	w3 = 2e-2

	l1 = torch.mean(torch.abs(y_true - y_pred))
	l2 = torch.mean(torch.abs(y_true_2spec - y_pred_2spec))
	l3 = fnorm(y_true_2spec - y_pred_2spec) / fnorm(y_true_2spec)
	
	l = w1*l1 + w2*l2 + w3*l3
	return l

@pit_wrapper_loss
def pit_mae_loss(y_pred, y_true):
	return mae_loss(y_pred, y_true)

@pit_wrapper_loss
def pit_raw2spec_mae_loss(y_pred, y_true, stft):
	return raw2spec_mae_loss(y_pred, y_true, stft)

@pit_wrapper_loss
def pit_mse_loss(y_pred, y_true):
	return mse_loss(y_pred, y_true)

@pit_wrapper_loss
def pit_raw2spec_mse_loss(y_pred, y_true, stft):
	return raw2spec_mse_loss(y_pred, y_true, stft)

@pit_wrapper_loss
def pit_spectral_convergence_loss(y_pred, y_true):
	return spectral_convergence_loss(y_pred, y_true)

@pit_wrapper_loss
def pit_raw2spec_spectral_convergence_loss(y_pred, y_true, stft):
	return raw2spec_spectral_convergence_loss(y_pred, y_true, stft)

@pit_wrapper_loss
def pit_neg_si_sdr(y_pred, y_true, zero_mean=False, epsilon=1e-10):
	return neg_si_sdr(y_pred, y_true, zero_mean=zero_mean, epsilon=epsilon)

@pit_wrapper_loss
def pit_total_loss(y_pred, y_true, stft):
	return total_loss(y_pred, y_true, stft)
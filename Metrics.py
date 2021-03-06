import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import *

def si_sdr(estimation, data, index=1, epsilon=1e-10):
	reference = data[index].detach()
	reference_energy = torch.sum(torch.pow(reference, 2), dim=-1, keepdim=True) + epsilon
	opt_scaling = torch.true_divide(torch.sum(torch.mul(reference, estimation), dim=-1, keepdim=True), reference_energy)
	projection = torch.mul(opt_scaling, reference)
	noise = torch.sub(estimation, projection)
	ratio = torch.true_divide(torch.sum(torch.pow(projection, 2), dim=-1), torch.sum(torch.pow(noise, 2), dim=-1) + epsilon)
	return torch.mean(torch.mul(torch.log10(ratio + epsilon), 10))

def accuracy(outputs, data, index=1, classifier=None):
	y_true = data[index].detach()
	try:
		assert outputs.ndim==2
	except:
		outputs = outputs.view(-1, outputs.size(-1))
		y_true = y_true.view(-1)
	if classifier is not None:
		classifier = classifier.to(outputs.device)
		outputs = classifier(outputs)
	_, pred = torch.max(outputs, dim = 1)
	correct = (pred == y_true).sum()
	total = y_true.size(0)
	acc = 100 * correct.item() / total
	return torch.tensor(acc, dtype=torch.float64)

def normalized_accuracy(outputs, data, index=1, classifier=None, peak_accuracy=100):
	y_true = data[index].detach()
	try:
		assert outputs.ndim==2
	except:
		outputs = outputs.view(-1, outputs.size(-1))
		y_true = y_true.view(-1)
	if classifier is not None:
		classifier = classifier.to(outputs.device)
		outputs = classifier(outputs)
	_, pred = torch.max(outputs, dim = 1)
	correct = (pred == y_true).sum()
	total = y_true.size(0)
	acc = 100 * correct / total
	acc = (1 - (F.relu(peak_accuracy - acc) / peak_accuracy)).item()
	return torch.tensor(acc, dtype=torch.float64)

@pit_wrapper_metric
def pit_si_sdr(y_pred, data, index):
	return si_sdr(y_pred, data, index)

@pit_wrapper_metric
def pit_accuracy(outputs, data, index, classifier=None):
	return accuracy(outputs, data, index, classifier=classifier)

@pit_wrapper_metric
def pit_probnorm_accuracy(outputs, data, index, classifier=None, peak_accuracy=100):
	return normalized_accuracy(outputs, data, index, classifier=classifier, peak_accuracy=peak_accuracy)
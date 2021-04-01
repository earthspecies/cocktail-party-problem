import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
import tqdm
import time
import re

class Trainer(object):
	def __init__(self, model, optimizer, scheduler=None, loss_func=None, metric_func=None, verbose=0, device='cuda', dest=None, **kwargs):
		
		self.device = device
		
		self.model = model.to(device)
		
		if type(loss_func) == dict:
			self.loss_func = loss_func
		elif hasattr(loss_func, '__call__'):
			self.loss_func = {'Loss': loss_func}
			
		self.optimizer = optimizer
		self.scheduler = scheduler

		try:
			self.switcher = kwargs['optimizer_switcher_callback']
		except KeyError:
			self.switcher = None

		assert ((self.scheduler and self.switcher) is None), 'Scheduler and switcher are incompatible options'

		try:
			self.saver = kwargs['model_saver_callback']
		except KeyError:
			self.saver = None

		if dest is None: assert self.saver is None, 'If no destination is specified, the saver callback cannot be activated'
		
		if type(metric_func) == dict:
			self.metric_func = metric_func
		elif hasattr(metric_func, '__call__'):
			self.metric_func = {'Metric': metric_func}
		else:
			self.metric_func = None
		
		self.verbose = verbose
		
		for key in kwargs:
			if 'loss' in key:
				self.loss_func[key] = kwargs[key]
			if 'metric' in key:
				self.metric_func[key] = kwargs[key]

		try:
			self.multi_loss_weights = list(kwargs['weights'])
		except KeyError:
			self.multi_loss_weights = [1]
		
		try:
			self.regularizer = kwargs['L2_regularizer_callback']
			lambda_factor = self.regularizer['lambda']

			def L2_reg(model):
				return lambda_factor * sum(p.pow(2.0).sum() for p in model.parameters())

			self.loss_func['L2_reg'] = lambda *args: L2_reg(self.model)
			self.multi_loss_weights.append(1)
		except KeyError:
			pass
		
		assert len(self.multi_loss_weights) == len(self.loss_func), 'Unbalanced loss functions and weights'
		
		self.dest = dest
		if dest is not None:
			assert type(dest) == str
			if self.dest[-1] != '/':
				self.dest += '/'
			if not os.path.isdir(dest):
				os.mkdir(dest)
				if not os.listdir(dest):
					os.mkdir(self.dest + 'Figures')
					os.mkdir(self.dest + 'Training Logs')
					os.mkdir(self.dest + 'Models')
					os.mkdir(self.dest + 'Evaluation Logs')
					os.mkdir(self.dest + 'Results')
			else:
				print('Directory already exists! Do you wish to continue? (Y/N)')
				user_input = input()
				if user_input == 'Y':
					clear_output()
				else:
					raise ValueError('Choose a different name before proceeding.')
	
	def fit(self, train_loader, val_loader, epochs):
		loss_history_train = [[] for _ in self.loss_func]
		if len(self.loss_func) > 1:
			loss_history_train.append([]) 
		
		loss_history_val = [[] for _ in self.loss_func]
		if len(self.loss_func) > 1:
			loss_history_val.append([])
		
		if self.metric_func is not None:
			metric_history_train = [[] for _ in self.metric_func]
		
			metric_history_val = [[] for _ in self.metric_func]
		else:
			metric_history_train = None
			metric_history_val = None
		
		for epoch in range(epochs):
			running_loss_train = [0.0 for _ in self.loss_func]
			if len(self.loss_func) > 1:
				running_loss_train.append(0.0)

			running_loss_val = [0.0 for _ in self.loss_func]
			if len(self.loss_func) > 1:
				running_loss_val.append(0.0)

			if self.metric_func is not None:
				running_metric_train = [0.0 for _ in self.metric_func]
				running_metric_val = [0.0 for _ in self.metric_func]
			else:
				running_metric_train = None
				running_metric_val = None
			
			starttime = time.time()
			
			running_loss_train, running_metric_train = self.train_step(train_loader, 
																	   running_loss_train, 
																	   running_metric_train)
			
			running_loss_val, running_metric_val = self.validation_step(val_loader, 
																		running_loss_val, 
																		running_metric_val)
			if self.scheduler is not None:
				self.scheduler.step()

			if self.switcher is not None:
				if (epoch+1) == self.switcher['epoch']:
					print(f'Switcher callback activated >>>>> Old Optimizer: {self.optimizer}')
					self.optimizer = self.switcher['optimizer'](self.model)
					print(f'                            >>>>> New Optimizer: {self.optimizer}')
					time.sleep(1)

			if self.saver is not None:
				if ((epoch+1) >= self.saver['epoch']) and ((epoch+1) % self.saver['save_every'] == 0):
					file_name = self.dest + 'Models/' + f'saver_epoch{epoch+1}.pt'
					torch.save(self.model.state_dict(), file_name)
			
			endtime = int(np.round(time.time() - starttime, decimals=0))
			try:
				its = np.round(len(train_loader) / endtime, decimals=2)
			except ZeroDivisionError:
				its = np.round(len(train_loader) / (endtime+1e-100), decimals=2)
			
			clear_output()
			for history_i, loss_i in zip(loss_history_train, running_loss_train):
				history_i.append(loss_i)
			for history_i, loss_i in zip(loss_history_val, running_loss_val):
				history_i.append(loss_i)
			
			if self.metric_func is not None:
				for history_i, metric_i in zip(metric_history_train, running_metric_train):
					history_i.append(metric_i)
				for history_i, metric_i in zip(metric_history_val, running_metric_val):
					history_i.append(metric_i)
			
			print_statement = f'Epoch: {epoch+1} \n     >>>>> '
			counter = 0
			if len(self.loss_func)==1:
				for key, value in zip(self.loss_func.keys(), running_loss_train):
					counter += 1
					if counter < len(self.loss_func.keys()):
						print_statement += f'Train {key}: {np.round(value, decimals=5)} --- '
					else:
						print_statement += f'Train {key}: {np.round(value, decimals=5)} \n     >>>>> '
			else:
				for key, value in zip([*self.loss_func.keys(), 'Total Loss'], running_loss_train):
					counter += 1
					if counter <= len(self.loss_func.keys()):
						print_statement += f'Train {key}: {np.round(value, decimals=5)} --- '
					else:
						print_statement += f'Train {key}: {np.round(value, decimals=5)} \n     >>>>> '
			
			counter = 0
			if len(self.loss_func)==1:
				for key, value in zip(self.loss_func.keys(), running_loss_val):
					counter += 1
					if counter < len(self.loss_func.keys()):
						print_statement += f'Val {key}: {np.round(value, decimals=5)} --- '
					else:
						print_statement += f'Val {key}: {np.round(value, decimals=5)}'
			else:
				for key, value in zip([*self.loss_func.keys(), 'Total Loss'], running_loss_val):
					counter += 1
					if counter <= len(self.loss_func.keys()):
						print_statement += f'Val {key}: {np.round(value, decimals=5)} --- '
					else:
						print_statement += f'Val {key}: {np.round(value, decimals=5)}'
						
			if self.metric_func is not None:
				counter = 0 
				for key, value in zip(self.metric_func.keys(), running_metric_train):
					counter += 1
					if counter == 1:
						print_statement += f'\n     >>>>> '
					print_statement += f'Train {key}: {np.round(value, decimals=5)} --- '
				counter = 0 
				for key, value in zip(self.metric_func.keys(), running_metric_val):
					counter += 1
					if counter < len(self.metric_func.keys()):
						print_statement += f'Val {key}: {np.round(value, decimals=5)} --- '
					else:
						print_statement += f'Val {key}: {np.round(value, decimals=5)} '
			
			print_statement += f'\n     >>>>> {endtime}s: {its}it/s'
				
			if self.verbose==1:
				print(print_statement)
				
			elif self.verbose==2:
				print(print_statement)
				fig = self.training_curves(epochs, 
										   loss_history_train, 
										   loss_history_val, 
										   metric_history_train,
										   metric_history_val)

		if self.dest is not None:
			file_name = self.dest + 'Training Logs/table.csv'
			d = {'Epochs': np.arange(1, epochs+1)}
			
			loss_keys = list(self.loss_func.keys())
			loss_weights = self.multi_loss_weights
			loss_keys = [f'{w}{k}' for w, k in zip(loss_weights, loss_keys)] 
			if len(self.loss_func) > 1:
				loss_keys.append('Total Loss')
				
			for k, value in zip(loss_keys, loss_history_train):
				k = f'Train {k}'
				d[k] = value
			for k, value in zip(loss_keys, loss_history_val):
				k = f'Val {k}'
				d[k] = value
				
			if self.metric_func is not None:
				metric_keys = list(self.metric_func.keys())
				for k, value in zip(metric_keys, metric_history_train):
					k = f'Train {k}'
					d[k] = value
				for k, value in zip(metric_keys, metric_history_val):
					k = f'Val {k}'
					d[k] = value
				
			df = pd.DataFrame(d)
			df.to_csv(file_name, index=False) 
			try:
				fig.savefig(self.dest + 'Figures/training_curves.png')
			except:
				pass
			return df
		else:
			return None
				 
	def train_step(self, dataloader, running_loss, running_metric=None):
		for i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
			data = [d_i.to(self.device) for d_i in data]
			
			self.model.train()
			self.optimizer.zero_grad()
			outputs = self.model(data[0])
			loss = [self.loss_func[key](outputs, data[1])*w for key, w in zip(self.loss_func.keys(), self.multi_loss_weights)]
			total_loss = sum(loss)
			total_loss.backward()
			self.optimizer.step()
			running_loss[:-1] = [r_i + l_i.item() for r_i, l_i in zip(running_loss, loss)]
			running_loss[-1] += total_loss.item()
			
			if self.metric_func is not None:
				metric = [self.metric_func[key](outputs.detach(), data) for key in self.metric_func.keys()]
				running_metric = [r_i + m_i.item() for r_i, m_i in zip(running_metric, metric)]
		
		running_loss = [r_i / len(dataloader) for r_i in running_loss]
		try:
			running_metric = [r_i / len(dataloader) for r_i in running_metric]
		except:
			running_metric = None
			
		return running_loss, running_metric  
	
	def validation_step(self, dataloader, running_loss, running_metric=None):
		for i, data in enumerate(dataloader):
			data = [d_i.to(self.device) for d_i in data]
			
			self.model.eval()
			with torch.no_grad():
				outputs = self.model(data[0])
				
				loss = [self.loss_func[key](outputs, data[1])*w for key, w in zip(self.loss_func.keys(), self.multi_loss_weights)]
				total_loss = sum(loss)
				running_loss[:-1] = [r_i + l_i.item() for r_i, l_i in zip(running_loss, loss)]
				running_loss[-1] += total_loss.item()

				if self.metric_func is not None:
					metric = [self.metric_func[key](outputs.detach(), data) for key in self.metric_func.keys()]
					running_metric = [r_i + m_i.item() for r_i, m_i in zip(running_metric, metric)]
		
		running_loss = [r_i / len(dataloader) for r_i in running_loss]
		
		try:   
			running_metric = [r_i / len(dataloader) for r_i in running_metric]
		except:
			running_metric = None
		return running_loss, running_metric
	
	def evaluate(self, dataloader, *args, to_device='cpu', return_data=True):
		loss_keys = list(self.loss_func.keys())
		if len(self.loss_func) > 1:
			loss_keys.append('Total Loss')

		running_loss = [0.0 for _ in self.loss_func]
		if len(self.loss_func) > 1:
			running_loss.append(0.0)

		if self.metric_func is not None:
			running_metric = [0.0 for _ in self.metric_func]
		else:
			running_metric = None

		self.model.eval()
		with torch.no_grad():
			self.model = self.model.to(to_device)
			for i, data_batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
				data_batch = [d_i.to(to_device).detach() for d_i in data_batch]
				predictions_batch = [self.model(data_batch[0]).detach()]

				loss = [self.loss_func[key](predictions_batch[0], data_batch[1])*w for key, w in zip(self.loss_func.keys(), self.multi_loss_weights)]
				total_loss = sum(loss)
				running_loss[:-1] = [r_i + l_i.item() for r_i, l_i in zip(running_loss, loss)]
				running_loss[-1] += total_loss.item()

				if self.metric_func is not None:
					metric = [self.metric_func[key](predictions_batch[0], data_batch) for key in self.metric_func.keys()]
					running_metric = [r_i + m_i.item() for r_i, m_i in zip(running_metric, metric)]

				if return_data:
					if i == 0:  
						data = [d_i.to('cpu') for d_i in data_batch]
						predictions = [p_i.to('cpu') for p_i in predictions_batch]
					else:
						data = [torch.cat([d_i.to('cpu'), db_i.to('cpu')], dim=0) for d_i, db_i in zip(data, data_batch)]
						predictions = [torch.cat([p_i.to('cpu'), pb_i.to('cpu')], dim=0) for p_i, pb_i in zip(predictions, predictions_batch)]

		final_loss = [r_i / len(dataloader) for r_i in running_loss]

		try:   
			final_metric = [r_i / len(dataloader) for r_i in running_metric]
		except:
			final_metric = None

		d = {}
		print_statement = f'Evaluation: \n     >>>>> '
		counter = 0
		for key, value in zip(loss_keys, final_loss):
			d[key] = value
			counter += 1
			if counter < len(self.loss_func.keys()):
				print_statement += f'{key}: {np.round(value, decimals=5)} --- '
			else:
				print_statement += f'{key}: {np.round(value, decimals=5)} '

		if self.metric_func is not None:
			counter = 0
			for key, value in zip(self.metric_func.keys(), final_metric):
				d[key] = value
				counter += 1
				if counter < len(self.metric_func.keys()):
					if counter == 1:
						print_statement += f'\n     >>>>> '
					print_statement += f'{key}: {np.round(value, decimals=5)} --- '
				else:
					if counter == 1:
						print_statement += f'\n     >>>>> '
					print_statement += f'{key}: {np.round(value, decimals=5)} '
		print(print_statement)
		if self.dest is not None:
			file_name = self.dest + 'Evaluation Logs/'
			for arg in args:
				file_name += f'{arg}_'
			file_name += 'table.csv'
			df = pd.DataFrame(d, index=[0])
			df.to_csv(file_name, index=False)
		if return_data:
			return data, predictions
		else:
			return None, None
	
	def predict(self, input, to_device='cpu'):
		self.model.eval()
		self.model = self.model.to(to_device)
		with torch.no_grad():
			output = self.model(input)
		return output
	
	def training_curves(self, iterations, loss_history_train, loss_history_val, metric_history_train=None, metric_history_val=None):

		if self.metric_func is not None:
			fig, axes = plt.subplots(1, 1 + len(self.metric_func), figsize=(12,4))
			fig.tight_layout(pad=3)
			
			if len(self.loss_func)==1:
				for key, value in zip(self.loss_func.keys(), loss_history_train):
					plot_range = np.arange(1, len(value)+1)
					axes[0].plot(plot_range, value, label=f'Train {key}')
				for key, value in zip(self.loss_func.keys(), loss_history_val):
					plot_range = np.arange(1, len(value)+1)
					axes[0].plot(plot_range, value, label=f'Val {key}')
			else:
				for key, value in zip([*self.loss_func.keys(), 'Total Loss'], loss_history_train):
					plot_range = np.arange(1, len(value)+1)
					axes[0].plot(plot_range, value, label=f'Train {key}')
				for key, value in zip([*self.loss_func.keys(), 'Total Loss'], loss_history_val):
					plot_range = np.arange(1, len(value)+1)
					axes[0].plot(plot_range, value, label=f'Val {key}')
					
			axes[0].set_xlabel('Epoch')
			axes[0].set_ylabel('Loss')
			axes[0].set_ylim(bottom=0)
			axes[0].set_xlim([1, iterations])
			#axes[0].set_xticks(np.arange(1, iterations+1))
			axes[0].legend()
			
			counter = 1
			for key, value in zip(self.metric_func.keys(), metric_history_train):
				plot_range = np.arange(1, len(value)+1)
				axes[counter].plot(plot_range, value, label=f'Train {key}')
				counter += 1
			counter = 1
			for key, value in zip(self.metric_func.keys(), metric_history_val):
				plot_range = np.arange(1, len(value)+1)
				axes[counter].plot(plot_range, value, label=f'Val {key}')
				counter += 1
			for ax in axes[1:]:
				ax.set_xlabel('Epoch')
				ax.set_ylabel('Metric')
				ax.set_xlim([1, iterations])
				#ax.set_xticks(np.arange(1, iterations+1))
				ax.legend()
			plt.show()
		else:
			fig = plt.figure(figsize=(6,4))
			if len(self.loss_func)==1:
				for key, value in zip(self.loss_func.keys(), loss_history_train):
					plot_range = np.arange(1, len(value)+1)
					plt.plot(plot_range, value, label=f'Train {key}')
				for key, value in zip(self.loss_func.keys(), loss_history_val):
					plot_range = np.arange(1, len(value)+1)
					plt.plot(plot_range, value, label=f'Val {key}')
			else:
				counter = 1
				for key, value in zip([*self.loss_func.keys(), 'Total Loss'], loss_history_train):
					plot_range = np.arange(1, len(value) + 1)
					plt.plot(plot_range, value, label=f'Train {key}')
				for key, value in zip([*self.loss_func.keys(), 'Total Loss'], loss_history_val):
					plot_range = np.arange(1, len(value) + 1)
					plt.plot(plot_range, value, label=f'Val {key}')      
			plt.xlabel('Epoch')
			plt.ylabel('Loss')
			plt.xlim([1, iterations])
			plt.ylim(bottom=0)
			#plt.xticks(np.arange(1, iterations+1))
			plt.legend()
			plt.show()
		return fig
	
	def save_model(self, *args, dir_path ='Models/'):
		if self.dest is None:
			file_name = ''
		else:
			file_name = self.dest
		
		file_name += dir_path
		
		loss_keys = list(self.loss_func.keys())
		if len(self.loss_func) > 1:
			loss_keys.append('Total Loss')
		loss_weights = self.multi_loss_weights
		for k, w in zip(loss_keys, loss_weights):
			k = re.sub('_', '', k)
			file_name += f'{w}{k}_'
			
		if self.metric_func is not None:
			metric_keys = list(self.metric_func.keys())
			for k in metric_keys:
				k = re.sub('_', '', k)
				file_name += f'{k}_'
		
		
		for i, arg in enumerate(args):
			if i < len(args)-1:
				file_name += f'{arg}_'
			else:
				file_name += f'{arg}'
		file_name += '.pt'
		
		if not os.path.exists(file_name):
			torch.save(self.model.state_dict(), file_name)
		else:
			print('File already exists! Do you wish to replace it? (Y/N)')
			replace = input()
			if replace == 'Y':
				torch.save(self.model.state_dict(), file_name)
			else:
				raise ValueError('Choose a different name or delete the existing file.')
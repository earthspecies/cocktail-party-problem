import json
import os
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--animal', type=str,
						help='animal root directory for config file')
	parser.add_argument('-f', '--file', type=str,
						help='name of config file')
	args = parser.parse_args()

	root = args.animal
	assert root in ['Macaque', 'Dolphin', 'Bat']
	if not os.path.isdir(root):
		os.mkdir(root)

	if root[-1] != r'/':
		root += r'/'

	file_name = args.file
	if file_name[-5:] != '.json':
		file_name += '.json'

	if args.animal == 'Macaque':

		config = {
			"data_preprocessing": {
				"general": {
					"n_src":2,
					"balance":False,
					"n_individuals":None,
					"n_open":None,
					"zipfile_save":True
				},
				"classifier": {
					"augmentation_factor":2,
					"shift_factor":0.6,
					"padding_scheme":"zero",
					"padding_side":"front"
				},
				"separator": {
					"mixing_to_memory": {
						"n_src":2,
						"training_size":12000,
						"validation_size":3000,
						"shift_factor":0.15,
						"shift_overlaps":True,
						"padding_scheme":"zero",
						"padding_side":"front"
					}
				}    
			},
			"classifier_dataset_params": {
				"n_src": 2,
				"objective": "classification",
				"stft_params": None,
				"filter_params": None,
				"shuffle": True
			},
			"classifier_learning_params": {
				"batch_size": 32,
				"learning_rate": 0.0003,
				"epochs": 100
			},
			"classifier_model_params": {
				"in_size": [
					None,
					1,
					23156
				],
				"n_classes": 8,
				"n_blocks": 4,
				"pool_size": 4,
				"input_mode": "raw",
				"stft_params": {
					"kernel_size": 1024,
					"stride": 64
				},
				"lin_dim":128,
				"dropout":0.25
			},
			"classifier_trainer_params": {
				"loss_func": {
					"nll_loss": "nll_weighted"
				},
				"metric_func": {
					"accuracy": "classifier_acc"
				},
				"verbose": 1,
				"device": "cuda",
				"dest": "Classifier",

				"model_saver_callback": {
						"epoch": 30,
						"save_every": 1
				}
			},
			"separator_dataset_params": {
				"n_src": 2,
				"objective": "separation",
				"stft_params": None,
				"filter_params": None
			},
			"separator_learning_params": {
				"batch_size": 16,
				"learning_rate": 0.001,
				"epochs": 100,
				"optimizer": "sgd",
				"momentum": 0.6
			},
			"separator_model_params": {
				"in_size": [
					None,
					1,
					23156
				],
				"n_src": 2,
				"n_blocks": 4,
				"pool_size":2,
				"batch_norm": True,
				"filterbank_params": {
					"nfft": 1024,
					"hop": 64
				},
				"input_mode": "stft",
				"output_mode": "istft",
				"phase_channel": False
			},
			"separator_trainer_params": {
				"loss_func": {
					"pit_multi_loss": "pit_total"
				},
				"metric_func": {
					"si_sdr": "pit_sisdr"
				},
				"verbose": 1,
				"device": "cuda",
				"dest": "Separator1",
				"params": {
					"accuracy_metric": "pit_separator_acc",
					"probnorm_acc_metric": "pit_probnorm_acc",
					"optimizer_switcher_callback": {
						"optimizer": "adamw_amsgrad",
						"learning_rate": 0.0003,
						"epoch": 3
					},
					"model_saver_callback": {
						"epoch": 35,
						"save_every": 1
					}
				}
			},
			"eval_return_data":False
		}


	elif args.animal == 'Dolphin':

		config = {
			"data_preprocessing": {
				 "general": {
					 "n_src":2,
					 "balance":False,
					 "n_individuals":8,
					 "n_open":None,
					 "zipfile_save":True
				 },
				 "classifier": {
					"augmentation_factor":16,
					"shift_factor":0.75,
					"padding_scheme":"zero",
					"padding_side":"front"
				},
				"separator": {
					"mixing_to_memory": {
						"n_src":2,
						"training_size":8000,
						"validation_size":2000,
						"shift_factor":0.1,
						"shift_overlaps":True,
						"padding_scheme":"zero",
						"padding_side":"front"
					}
				}    
			},
			"classifier_dataset_params": {
				"n_src": 2,
				"objective": "classification",
				"stft_params": None,
				"filter_params": None,
				"shuffle": True
			},
			"classifier_learning_params": {
				"batch_size": 8,
				"learning_rate": 0.0003,
				"epochs": 50
			},
			"classifier_model_params": {
				"in_size": [
					None,
					1,
					290680
				],
				"n_classes": 8,
				"n_blocks": 4,
				"pool_size": 4,
				"input_mode": "raw",
				"stft_params": {
					"kernel_size": 1024,
					"stride": 256
				},
				"filter_params": {
					"cutoff_freq": 4700,
					"sample_rate": 96000,
					"b": 0.08
				},
				"lin_dim":128,
				"dropout":0.5
			},
			"classifier_trainer_params": {
				"loss_func": {
					"nll_loss": "nll"
				},
				"metric_func": {
					"accuracy": "classifier_acc"
				},
				"verbose": 1,
				"device": "cuda",
				"dest": "Classifier",
				
				"model_saver_callback": {
						"epoch": 15,
						"save_every": 1
				}
			},
			"separator_dataset_params": {
				"n_src": 2,
				"objective": "separation",
				"stft_params": None,
				"filter_params": {
					"cutoff_freq": 4700,
					"sample_rate": 96000,
					"b": 0.08
				}
			},
			"separator_learning_params": {
				"batch_size": 8,
				"learning_rate": 0.001,
				"epochs": 100,
				"optimizer": "sgd",
				"momentum": 0.6
			},
			"separator_model_params": {
				"in_size": [
					None,
					1,
					290680
				],
				"n_src": 2,
				"n_blocks": 3,
				"pool_size":6,
				"batch_norm": True,
				"filterbank_params": {
					"nfft": 1024,
					"hop": 256
				},
				"input_mode": "stft",
				"output_mode": "istft",
				"phase_channel": False
			},
			"separator_trainer_params": {
				"loss_func": {
					"pit_multi_loss": "pit_total"
				},
				"metric_func": {
					"si_sdr": "pit_sisdr"
				},
				"verbose": 1,
				"device": "cuda",
				"dest": "Separator",
				"params": {
					"accuracy_metric": "pit_separator_acc",
					"probnorm_acc_metric": "pit_probnorm_acc",
					"optimizer_switcher_callback": {
						"optimizer": "adamw_amsgrad",
						"learning_rate": 0.0003,
						"epoch": 3
					},
					"model_saver_callback": {
						"epoch": 35,
						"save_every": 1
					}
				}
			},
			"eval_return_data":False
		}

	elif args.animal == 'Bat':

		config = {
			"data_preprocessing": {
				 "general": {
					 "n_src":2,
					 "balance":False,
					 "n_individuals":None,
					 "n_open":3,
					  "zipfile_save":False
				 },
				 "classifier": {
					"augmentation_factor":1,
					"shift_factor":0.15,
					"padding_scheme":"zero",
					"padding_side":"front"
				},
				"separator": {
					"mixing_to_memory": None,
					"mixing_from_disk": {
						"n_src":2,
						"training_size":24000,
						"validation_size":6000,
						"shift_factor":0.1,
						"shift_overlaps":False,
						"padding_scheme":"zero",
						"padding_side":"front"
					}
				}    
			},
			"classifier_dataset_params": {
				"n_src": 2,
				"objective": "classification",
				"stft_params": None,
				"filter_params": None,
				"shuffle": True
			},
			"classifier_learning_params": {
				"batch_size": 8,
				"learning_rate": 0.0003,
				"epochs": 100
			},
			"classifier_model_params": {
				"in_size": [
					None,
					1,
					250000
				],
				"n_classes": 12,
				"n_blocks": 4,
				"pool_size": 4,
				"input_mode": "raw",
				"stft_params": {
					"kernel_size": 2048,
					"stride": 512
				},
				"lin_dim":128,
				"dropout":0.5
			},
			"classifier_trainer_params": {
				"loss_func": {
					"nll_loss": "nll_weighted"
				},
				"metric_func": {
					"accuracy": "classifier_acc"
				},
				"verbose": 1,
				"device": "cuda",
				"dest": "Classifier",
				
				"model_saver_callback": {
						"epoch": 30,
						"save_every": 1
				}
			},
			"separator_dataset_params": {
				"n_src": 2,
				"objective": "separation",
				"stft_params": None,
				"filter_params": None
			},
			"separator_learning_params": {
				"batch_size": 8,
				"learning_rate": 0.001,
				"epochs": 100,
				"optimizer": "sgd",
				"momentum": 0.6
			},
			"separator_model_params": {
				"in_size": [
					None,
					1,
					250000
				],
				"n_src": 2,
				"n_blocks": 4,
				"pool_size": 3,
				"batch_norm": True,
				"filterbank_params": {
					"nfft": 2048,
					"hop": 512
				},
				"input_mode": "stft",
				"output_mode": "istft",
				"phase_channel": False
			},
			"separator_trainer_params": {
				"loss_func": {
					"pit_multi_loss": "pit_total"
				},
				"metric_func": {
					"si_sdr": "pit_sisdr"
				},
				"verbose": 1,
				"device": "cuda",
				"dest": "Separator",
				"params": {
					"accuracy_metric": "pit_separator_acc",
					"probnorm_acc_metric": "pit_probnorm_acc",
					"optimizer_switcher_callback": {
						"optimizer": "adamw_amsgrad",
						"learning_rate": 0.0003,
						"epoch": 3
					},
					"model_saver_callback": {
						"epoch": 35,
						"save_every": 1
					},
					"L2_regularizer_callback": {
						"lambda":0.000001
					}
				}
			},
			"eval_return_data":False
		}
	with open(root+file_name, 'w') as fp:
		json.dump(config, fp, indent=4)
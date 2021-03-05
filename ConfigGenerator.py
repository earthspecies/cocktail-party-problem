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
			"classifier_dataset_params": {
				"n_src": 2,
				"objective": "classification",
				"stft_params": None,
				"filter_params": None,
				"shuffle":True
			},
			"classifier_learning_params": {
				"batch_size": 32,
				"learning_rate": 0.0001,
				"epochs": 18
			},
			"classifier_model_params": {
				"in_size": [
					None,
					1,
					23156
				],
				"n_classes": 8,
				"n_blocks":6,
				"pool_size":2,
				"input_mode": "raw",
				"stft_params": {
					"kernel_size": 1024,
					"stride": 64
				}
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
				"dest": "Classifier"
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
				"epochs": 85,
				"optimizer": "adamw"
				#"optimizer": "sgd",
				#"momentum": 0.6
			},
			"separator_model_params": {
				"in_size": [
					None,
					1,
					23156
				],
				"n_src": 2,
				"n_blocks": 4,
				"batch_norm": True,
				"filterbank_params": {
					"nfft": 1024,
					"hop": 64
				},
				"input_mode": "conv1d",
				"output_mode": "conv1d",
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
				"dest": "Separator0",
				"params": {
					"accuracy_metric": "pit_separator_acc",
					#"optimizer_switcher_callback": {
						#"optimizer": "adamw_amsgrad",
						#"learning_rate": 0.0003,
						#"epoch": 3
					#},
					"model_saver_callback": {
						"epoch": 40,
						"save_every": 1
					}
				}
			}
		}

	elif args.animal == 'Dolphin':

		config = {
			"classifier_dataset_params": {
				"n_src": 2,
				"objective": "classification",
				"stft_params": None,
				"filter_params": None,
				"shuffle":True
			},
			"classifier_learning_params": {
				"batch_size": 8,
				"learning_rate": 0.0001,
				"epochs": 8
			},
			"classifier_model_params": {
				"in_size": [
					None,
					1,
					290680
				],
				"n_classes": 8,
				"n_blocks":8,
				"pool_size":2,
				"input_mode": "raw",
				"stft_params": {
					"kernel_size": 1024,
					"stride": 1024 // 4
				}
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
				"dest": "Classifier"
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
				"epochs": 45,
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
				"n_blocks": 4,
				"batch_norm": True,
				"filterbank_params": {
					"nfft": 1024,
					"hop": 1024 // 4
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
					"optimizer_switcher_callback": {
						"optimizer": "adamw_amsgrad",
						"learning_rate": 0.0003,
						"epoch": 3
					},
					"model_saver_callback": {
						"epoch": 20,
						"save_every": 1
					}
				}
			}
		}

	elif args.animal == 'Bat':

		config = {
			"classifier_dataset_params": {
				"n_src": 2,
				"objective": "classification",
				"stft_params": None,
				"filter_params": None,
				"shuffle":True
			},
			"classifier_learning_params": {
				"batch_size": 8,
				"learning_rate": 0.0001,
				"epochs": 8
			},
			"classifier_model_params": {
				"in_size": [
					None,
					1,
					200000
				],
				"n_classes": 7,
				"n_blocks":8,
				"pool_size":2,
				"input_mode": "raw",
				"stft_params": {
					"kernel_size": 1024,
					"stride": 1024 // 4
				}
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
				"dest": "Classifier"
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
				"epochs": 45,
				"optimizer": "sgd",
				"momentum": 0.6
			},
			"separator_model_params": {
				"in_size": [
					None,
					1,
					200000
				],
				"n_src": 2,
				"n_blocks": 4,
				"batch_norm": True,
				"filterbank_params": {
					"nfft": 1024,
					"hop": 1024 // 4
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
					"optimizer_switcher_callback": {
						"optimizer": "adamw_amsgrad",
						"learning_rate": 0.0003,
						"epoch": 3
					},
					"model_saver_callback": {
						"epoch": 20,
						"save_every": 1
					}
				}
			},
			"eval_return_data": False
		}
	with open(root+file_name, 'w') as fp:
		json.dump(config, fp, indent=4)
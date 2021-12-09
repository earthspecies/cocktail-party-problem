import pandas as pd
import re
import os
try:
	from fastai.vision.all import untar_data, get_files
except:
	from fastai2.vision.all import untar_data, get_files
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import random
import tqdm
from Utils import id_mapper

class LoadMacaqueData(object):    
	def __init__(self, url='https://storage.googleapis.com/ml-bioacoustics-datasets/macaques_24414Hz.zip', sr=24414, os='Ubuntu', state=0):
		self.url = url
		self.os = os
		self.state = state
		self.sr = sr
		
	def construct_dataframe(self):
		path = untar_data(self.url)
		wav_files = sorted(get_files(path))

		wfs = []
		labels = []
		wavs = []

		if self.os == 'Windows':
			pattern = r'\\'
		else:
			pattern ='/'

		for wav in wav_files:
			call_code = re.split(pattern, str(wav))[-2]
			wf, _ = sf.read(wav)

			wfs.append(wf)
			labels.append(call_code)
			wavs.append(wav)

		call_dict = {l: i for i, l in enumerate(np.unique(labels))}
		call_category = [call_dict[i] for i in labels]
		data_df = pd.DataFrame({'Waveform':wfs, 'Path': wavs, 'Label': labels, 'Category': call_category})
		
		return data_df
	
	def fixed_dataframe(self):
		
		dataframe = self.construct_dataframe()
		
		mean_dur = self.get_mean_duration(dataframe)
		std_dur = self.get_std_duration(dataframe)
		paths = dataframe.Path.values
		labels = dataframe.Label.values
		categories = dataframe.Category.values
		waveforms = dataframe.Waveform.values

		xs = []
		for wf in waveforms:
			x = librosa.util.fix_length(wf, mean_dur + 3*std_dur)
			xs.append(x)
		fixed_df = pd.DataFrame({'Waveform':xs, 'Path': paths, 'Label':labels, 'Category':categories})    
		return fixed_df
	
	def balanced_dataframe(self):
	
		dataframe = self.fixed_dataframe()
		balanced_df = dataframe.groupby('Category')
		balanced_df = balanced_df.apply(lambda x: x.sample(balanced_df.size().min(),
														   random_state=self.state).reset_index(drop=True))

		return balanced_df
	
	def visualize_classes(self, dataframe, group='Label'):
		df = dataframe.groupby(group).apply(lambda x: x.sample(1, random_state=self.state))
		df = df.reset_index(drop=True)

		fig, axes = plt.subplots(2,4, figsize=(15,10))
		for i, ax in enumerate(axes.flatten()):
			ax.plot(df.Waveform.iloc[i], linewidth=0.4)
			ax.set_title(df.Label.iloc[i])
		plt.show()

	def run(self, balance=False):
		if balance:
			data_df = self.balanced_dataframe()
		else:
			data_df = self.fixed_dataframe()
		mean_dur = self.get_mean_duration(data_df)
		std_dur = self.get_std_duration(data_df)

		win_width = mean_dur + 3*std_dur

		X = []
		for i, wf in enumerate(data_df.Waveform.values):
			X.append(wf.astype('float32'))
		Y = data_df.Category.values.astype('int64').tolist()
		return X, Y
	
	@staticmethod
	def get_mean_duration(dataframe):
		waveforms = dataframe.Waveform.values
		durs = []
		for wf in waveforms:
			durs.append(wf.shape[0])
		mean_dur = int(np.mean(durs))
		return mean_dur

	@staticmethod
	def get_std_duration(dataframe):
		waveforms = dataframe.Waveform.values
		durs = []
		for wf in waveforms:
			durs.append(wf.shape[0])
		std_dur = int(np.std(durs))
		return std_dur

class LoadDolphinData(object):
	def __init__(self, os='Ubuntu', sr=96000, n_individuals=None, frames_fx=np.max, seed=1234):#V0_42):
		self.os = os
		if os == 'Windows':
			self.data_path = 'BioacousticData\\Dolphin'
		else:
			self.data_path = 'BioacousticData/Dolphin'
		self.sr = sr 
		self.n_individuals = n_individuals
		self.frames_fx = frames_fx
		self.seed = seed
		random.seed(seed)
        
	def load_wavs(self):
		wavs = []
		for r, d, f in os.walk(self.data_path):
			for item in f:
				if '.wav' in item:
					wavs.append(os.path.join(r, item))
		wavs.sort()
		return wavs
	
	def generate_df(self):
		
		paths = []
		
		ids = []
		classes = []
		durs = []
		
		wav_files = self.load_wavs()
		IDs = self.get_ids(wav_files)
		id_dict = {ID:i for i, ID in enumerate(IDs)}

		for f in wav_files:
			paths.append(f)

			ID = self.get_id(f)
			ids.append(ID)
			classes.append(id_dict[ID])

			durs.append(sf.info(f).duration)
			
		df = pd.DataFrame({'Class ID':classes, 'Dolphin ID': ids, 'Wav Path': paths, 'Duration (s)': durs})
		
		if self.n_individuals is not None:
			assert type(self.n_individuals) == int

			id_list = df['Class ID'].unique().tolist()
			n_total = len(id_list)
			assert self.n_individuals < n_total

			ids_to_remove = random.sample(id_list, n_total - self.n_individuals)
			df = df[~df['Class ID'].isin(ids_to_remove)]
		return df
	
	def run(self):
		path_df = self.generate_df()
		frames = int(self.frames_fx(path_df['Duration (s)'].values)*self.sr)
		X, Y = [], []
		for i in range(len(path_df)):
			wav, _ = librosa.load(path_df['Wav Path'].iloc[i], sr=self.sr)
			wav = librosa.util.fix_length(wav, frames).astype('float32')
			X.append(wav)
			Y.append(path_df['Class ID'].iloc[i])
		if self.n_individuals is not None:
			Y = id_mapper(Y)    
		return X, Y
	
	def get_ids(self, files):
		ids = []
		for f in files:
			ID = self.get_id(f)
			if ID not in ids:
				ids.append(ID)
		return ids
	
	@staticmethod
	def get_id(file_name):
		pattern = 'FB\d+'
		match = re.findall(pattern, file_name)[0]
		return match

class LoadBatData(object):
	def __init__(self, os='Ubuntu', sr=250000, frames=250000):
		self.os = os
		if os=='Windows':
			self.pattern = r'\\'
		else:
			self.pattern = r'/'
		self.sr = sr
		self.frames = frames
		self.base_path = f'BioacousticData{self.pattern}EgyptianFruitBat{self.pattern}'

	def load_wavs(self):
		wavs = []
		file_names = []
		for r, d, f in os.walk(self.base_path):
			for item in f:
				print
				if '.WAV' in item:
					wavs.append(os.path.join(r, item))
					fn = re.split(self.pattern, item)[-1]
					file_names.append(fn)
		return wavs, file_names

	def generate_df(self, csv_file='better_annotations.csv', count_threshold=1000, balance=False):
		wavs, file_names = self.load_wavs()

		csv_path = self.base_path + csv_file
		df = pd.read_csv(csv_path)
		df = df[df['File name'].isin(file_names)]

		idxs = df[df['Emitter']<=0].index 
		df.drop(idxs , inplace=True)

		value_counts = df['Emitter'].value_counts()
		to_remove = value_counts[value_counts <= count_threshold].index
		df = df[~df.Emitter.isin(to_remove)]

		if balance:
			df = df.groupby('Emitter')
			df = df.apply(lambda x: x.sample(df.size().min(),random_state=42).reset_index(drop=True))

		drop_list = [
			 'Unnamed: 0',
			 'FileID',
			 'Addressee', 
			 'Context', 
			 'Emitter pre-vocalization action',
			 'Emitter post-vocalization action',
			 'Addressee post-vocalization action',
			 'Addressee pre-vocalization action'
		]
		for d in drop_list:
			df=df.drop(d , axis='columns')
		df = df.reset_index(drop=True)
		return df
	
	def run(self, balance=False, chunk='random', start_offset_from_end=1.5):
		df = self.generate_df(balance=balance)
		
		id_dict = {e:i for i, e in enumerate(np.unique(df.Emitter.values))}
		X, Y = [], []

		if chunk == 'random':
			for i in tqdm.tqdm(range(len(df))):
				folder = df['File folder'].iloc[i]
				fname = df['File name'].iloc[i]
				path = self.base_path
				path += f'{folder}{self.pattern}{fname}'
				end_sample = df['End sample'].iloc[i]
				start_sample = df['Start sample'].iloc[i]
				start_offset = random.randint(start_sample, 
											  max(start_sample + 1, end_sample - start_offset_from_end*self.frames))
				x, _ = librosa.load(path, 
									offset=start_offset/self.sr, 
									duration=self.frames/self.sr, 
									sr=None)
				x = librosa.util.fix_length(x, self.frames).astype('float32')
				X.append(x)
				Y.append(id_dict[df['Emitter'].iloc[i]])
		
		elif chunk == 'segment':
			for i in tqdm.tqdm(range(len(df))):

				sample_start = df['Start sample'].iloc[i]
				sample_end = df['End sample'].iloc[i]
			
				start_frame = df['Start sample'].iloc[i]
				end_frame = start_frame + self.frames
			
				folder = df['File folder'].iloc[i]
				fname = df['File name'].iloc[i]
				path = self.base_path
				path += f'{folder}{self.pattern}{fname}'

				while end_frame < sample_end:

					x, _ = librosa.load(path, 
										offset=start_frame/self.sr, 
										duration=self.frames/self.sr, 
										sr=None)
					x = librosa.util.fix_length(x, self.frames).astype('float32')
					
					X.append(x)
					Y.append(id_dict[df['Emitter'].iloc[i]])

					start_frame += self.frames
					end_frame += self.frames

		return X, Y
    
class LoadSpermWhaleData(object):
	def __init__(self, os='Ubuntu', frames=264000, count_threshold=300):
		self.os = os
		if os=='Windows':
			self.pattern = r'\\'
		else:
			self.pattern = '/'
		self.data_path = f'BioacousticData{self.pattern}SpermWhale'
		self.frames = frames
		self.count_threshold = count_threshold
	
	def generate_df(self, balance=False):
		anno_df = pd.read_csv(f'{self.data_path}{self.pattern}all_codas_with_frames.csv')
		anno_df = anno_df.drop(anno_df[(anno_df.IDN == 0) | (anno_df.IDN == 9999)].index)
		anno_df = anno_df[anno_df.groupby('IDN')['IDN'].transform('count').ge(self.count_threshold)]
		if balance:
			balanced_df = anno_df.groupby('IDN')
			anno_df = balanced_df.apply(lambda x: x.sample(balanced_df.size().min(), random_state=42).reset_index(drop=True))
		return anno_df
	
	def run(self, balance=False):
		df = self.generate_df(balance=balance)
		id_dict = {ID: i for i, ID in enumerate(np.unique(df.IDN.values))}

		X, Y = []
		for i in range(len(df)):
			X.append(self.read_coda(df.iloc[i]).astype('float32'))
			Y.append(id_dict[df.IDN.iloc[i]])
		return X, Y
		
	def read_coda(self, row):
		data, _ = sf.read(f'{self.data_path}{self.pattern}{row.filename}', frames=self.frames, start=row.start_frame-12000)
		data_T = data.T
		return data_T[0]

class LoadElephantData(object):
	def __init__(self, os='Ubuntu', sr=2000, frames=8000):
		self.os = os
		if os=='Windows':
			self.pattern = r'\\'
		else:
			self.pattern = r'/'
		self.sr = sr
		self.frames = frames

		self.data_path = f'BioacousticData{self.pattern}Elephant{self.pattern}'
	
	def run(self):
		audio_path = self.data_path + f'audio{self.pattern}'
		anno_path = self.data_path + 'annotations.csv'

		df = pd.read_csv(anno_path)
		df = df.drop(df[(df.Callers == 'Emma') | (df.Callers == 'Erin') | (df.Callers == 'Eudora') | (df.Callers == 'Enid')].index)
		
		X, Y = [], []
		for i, name in enumerate(df['Callers'].unique()):
			id_df = df[df['Callers'] == name]
			for j, row in id_df.iterrows():
				file = row.SndFile
				file_name = audio_path + f'{file}.wav'
				
				info = sf.info(file_name)
				end_time = info.duration
				segments = int(end_time * self.sr) // self.frames
				if segments == 0:
					segments = 1
				
				start_time=0
				for s in range(segments):
					x, _ = librosa.load(file_name, 
										offset=start_time, 
										duration=self.frames/self.sr, 
										sr=self.sr)
					start_time += self.frames/self.sr
					x = librosa.util.fix_length(x, self.frames).astype('float32')
					X.append(x)
					Y.append(i)
		return X, Y

import torch
import librosa
import numpy as np
import librosa.display
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ipywidgets import Button, interact, interactive, fixed, interact_manual
from IPython.display import display, Audio, clear_output, Image
from matplotlib.gridspec import GridSpec
from scipy.io import wavfile
import ipywidgets as widgets
import functools
import glob
import re

from Layers import HighPassFilter

def ax_format(x, pos):
	x /= 1000
	if float(x).is_integer():
		return f'{int(x)}'
	else:
		return f'{round(x, 1)}'

y_format = tkr.FuncFormatter(ax_format)

def compute_spectrogram(signal, nfft, hop, ref=np.max):
	S = librosa.stft(signal, n_fft=nfft, hop_length=hop)
	D = librosa.amplitude_to_db(np.abs(S), ref=ref)
	return D

def plot_separations(X, Y, index, model, sr, domain='time-domain', dest=None, playback_factor=1, **kwargs):
	if dest is not None:
		if dest[-1] != r'/':
			dest += r'/'

	window = X.size(-1)
	n_src = Y.size(1)
	
	t = np.linspace(0, window/sr, window)
	X_input = torch.unsqueeze(X[index, :, :], dim=0)
    
	out = model(X_input).detach().numpy()
	outs = [out[0, s, :] for s in range(n_src)]
    
	save_with_filter = ''
	if domain=='time-domain':
		try:
			filter_params = kwargs['filter_params']
			hpf = HighPassFilter(**filter_params)
			Ys = [hpf(Y[index, s, :].view(1, -1)).squeeze().numpy() for s in range(n_src)]
			#X_input = hpf(X_input.view(1, -1))
			save_with_filter = '_filtered'
		except KeyError:
			hpf = None
			Ys = [Y[index, s, :].numpy() for s in range(n_src)]
	else:
		Ys = [Y[index, s, :].numpy() for s in range(n_src)]

	audios = [np.squeeze(X_input.numpy()), *Ys, *outs]

	if domain == 'time-frequency-domain':
		stft_params = kwargs['stft_params']
		try:
			reference = kwargs['ref']
		except:
			reference = None
		if reference is None:
			S_ref = np.max(np.abs(librosa.stft(audios[0], 
											   n_fft=stft_params['nfft'], 
											   hop_length=stft_params['hop'])))
		else:
			S_ref = reference
		specs = [compute_spectrogram(a, 
									 stft_params['nfft'], 
									 stft_params['hop'],
									 ref=S_ref) for a in audios]
	Y_labels = [f'Source Ch. {i}' for i in range(n_src)]
	out_labels = [f'Separated Ch. {i}' for i in range(n_src)]
	labels = ['Mixture', *Y_labels, *out_labels]
	
	n_plots = 2*n_src + 1
	
	fig = plt.figure(constrained_layout=True, figsize=(16, 8))
	gs = GridSpec(2, 2 + n_src, figure=fig)
	ax_mix = fig.add_subplot(gs[:, 0:2])
	ax_mix.set_xlabel('Time (s)')
	
	ax_sources = [fig.add_subplot(gs[0, 2+j]) for j in range(n_src)]
	for ax in ax_sources:
		ax.set_xticks([])
		ax.set_xlabel('')
	ax_separateds = [fig.add_subplot(gs[1, 2+j]) for j in range(n_src)]
	for ax in ax_separateds:
		ax.set_xlabel('Time (s)')
	
	axes = [ax_mix, *ax_sources, *ax_separateds]
	
	for ax, l in zip(axes, labels):
		ax.set_title(l)
		
	if domain == 'time-domain':
		for i in range(n_plots):
			axes[i].plot(t, audios[i])
			if i == 0:
				axes[i].set_ylabel('Amplitude')
				ylims = axes[i].get_ylim()
			axes[i].set_title(labels[i])
			axes[i].set_ylim(ylims)
			if i not in [0, 1, 1 + n_src]:
				axes[i].set_ylabel('')
				axes[i].set_yticks([])
			else:
				axes[i].set_ylabel('Amplitude')

	elif domain =='time-frequency-domain':
		for i in range(n_plots):
			spec = librosa.display.specshow(specs[i], 
							 x_axis='time', 
							 y_axis='linear', 
							 sr=sr, 
							 hop_length=stft_params['hop'],
							 ax=axes[i],
							 cmap='magma',
							 vmin=np.min(specs[0]),
							 vmax=0) 
			axes[i].set_xlabel('Time (s)')
			axes[i].set_ylabel('Frequency (kHz)')
			axes[i].yaxis.set_major_formatter(y_format)
			try:
				fmax = kwargs['fmax']
				axes[i].set_ylim([0, fmax])
			except:
				pass
			try:
				cbar = kwargs['cbar']
				if cbar == 'multiple':
					fig.colorbar(spec, ax=axes[i])
				elif cbar == 'single':
					axins = inset_axes(axes[-1],
						width="5%",
						height="200%",
						loc='lower left',
						bbox_to_anchor=(1.1, 0.1, 1, 1),
						bbox_transform=axes[-1].transAxes,
						borderpad=0)
					fig.colorbar(spec, cax=axins, format='%+2.f dB')
			except:
				pass
			axes[i].set_title(labels[i])
			if i not in [0, 1, 1 + n_src]:
				axes[i].set_ylabel('')
				axes[i].set_yticks([])
			if i in list(range(1, n_src+1)):
				axes[i].set_xlabel('')
				axes[i].set_xticks([])
	if dest is not None:
		plt.savefig(f'{dest}Results/CPP_{domain}_Index{index}.png')
	plt.show()
    
	if dest is not None:
		for a, l in zip(audios, labels):
			if (l == 'Mixture') and (domain=='time-domain') and (hpf is not None):
				a = hpf(X_input.view(1, -1)).squeeze().numpy()
				audios[0] = a
			wavfile.write(f'{dest}Results/{l}_Index{index}{save_with_filter}.wav', sr, a)
			
	buttons_list = [widgets.Button(description=labels[i]) for i in range(n_plots)]
	for n in range(n_plots):
		buttons_list[n].style.button_color = 'lightgray'
	out = widgets.Output()
	def on_button_clicked(_, i=0):
		with out:
			clear_output()
			display(Audio(audios[i], rate=sr//playback_factor))
			for n in range(n_plots):
				buttons_list[n].style.button_color = 'lightgray'
			_.style.button_color = 'pink'
			
	button_click_fxs = [functools.partial(on_button_clicked, i=j) for j in range(1, n_plots)]
	button_click_fxs.insert(0, on_button_clicked)
	for k in range(n_plots):
		buttons_list[k].on_click(button_click_fxs[k])
	
	buttons = widgets.HBox(buttons_list)
	display(widgets.VBox([buttons, out]))
    
def audio_visual(animal, n_src, rep, playback_factor=1, directory='Assets'):
	assert animal in ['Macaque', 'Dolphin', 'Bat'], print('Animal must be Macaque, Dolphin, or Bat')
	assert rep in ['WF', 'TFR'], print('Rep must be WF or TFR')

	mixture_wav = sorted(glob.glob(f'{directory}/{animal}/{n_src}SpeakerMixture.wav'))
	source_wavs = sorted(glob.glob(f'{directory}/{animal}/{n_src}SpeakerSource*.wav'))
	pred_wavs = sorted(glob.glob(f'{directory}/{animal}/{n_src}SpeakerPred*.wav'))
	wavs = [*mixture_wav, *source_wavs, *pred_wavs]
	_, sr = librosa.load(wavs[0], sr=None)
	audios = [librosa.load(w, sr=None)[0] for w in wavs]
    
	plots = Image(f'{directory}/{animal}/{n_src}Speaker{rep}.png')
    
	labels = [re.findall('Speaker(\w+).wav', w)[0] for w in wavs]
	labels = [re.sub('Source', 'Source ', l) for l in labels]
	labels = [re.sub('Pred', 'Separated ', l) for l in labels]
    
	n_plots = 2*n_src+1
    
	buttons_list = [widgets.Button(description=labels[i]) for i in range(n_plots)]
	for n in range(n_plots):
		buttons_list[n].style.button_color = 'lightgray'
	out = widgets.Output()
	def on_button_clicked(_, i=0):
		with out:
			clear_output()
			display(Audio(audios[i], rate=sr//playback_factor))
			for n in range(n_plots):
				buttons_list[n].style.button_color = 'lightgray'
			_.style.button_color = 'pink'

	button_click_fxs = [functools.partial(on_button_clicked, i=j) for j in range(1, n_plots)]
	button_click_fxs.insert(0, on_button_clicked)
	for k in range(n_plots):
		buttons_list[k].on_click(button_click_fxs[k])

	buttons = widgets.HBox(buttons_list)
	display(plots, widgets.VBox([buttons, out]))
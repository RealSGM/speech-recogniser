import numpy as np
import os
import librosa
import audio.file_reader as file_reader

TRAINING_PATH = "data/audio/training_mfccs"
TESTING_PATH = "data/audio/testing_mfccs"
MIN_INT = 1e-10


def spectral_features(magnitude: np.ndarray, fs: int) -> np.ndarray:
	# Compute spectral centroid, bandwidth, flatness using librosa
	centroid = librosa.feature.spectral_centroid(S=magnitude[np.newaxis, :], sr=fs)[0, 0]
	bandwidth = librosa.feature.spectral_bandwidth(S=magnitude[np.newaxis, :], sr=fs)[0, 0]
	flatness = librosa.feature.spectral_flatness(S=magnitude[np.newaxis, :])[0, 0]

	centroid /= (fs / 2)
	bandwidth /= (fs / 2)
	flatness = np.clip(flatness, 0.0, 1.0)

	return np.array([centroid, bandwidth, flatness])


def pad_mfcc_features(data: list[np.ndarray], max_len: int, num_features: int) -> np.ndarray:
	padded = np.zeros((len(data), max_len, num_features))
	for i, m in enumerate(data):
		length = m.shape[0]
		padded[i, :length, :] = m
	return padded


def compute_deltas(mfcc_features: np.ndarray, N: int = 2) -> np.ndarray:
	return librosa.feature.delta(mfcc_features.T, order=1, width=2 * N + 1).T


def preprocess_signal(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
	emphasised = librosa.effects.preemphasis(signal, coef=coeff)
	return emphasised - np.mean(emphasised)


def mfcc(speechFrame: np.ndarray, fs: int, num_mel_filters: int, num_ceps: int,
		use_spectral_features: bool, use_mel_filter_bank: bool) -> np.ndarray:

	# Compute magnitude spectrum
	S = np.abs(librosa.stft(speechFrame, n_fft=len(speechFrame), window='hamming'))[:, 0]

	# Energy
	frame_energy = np.log(np.sum(S**2) + MIN_INT)

	# Triangular Mel or custom filterbank -> librosa handles Mel-spaced
	mfcc_vals = librosa.feature.mfcc(
		y=speechFrame,
		sr=fs,
		n_mfcc=num_ceps,
		n_mels=num_mel_filters,
		hop_length=len(speechFrame),	# single frame
		n_fft=len(speechFrame)
	).flatten()

	if use_spectral_features:
		spec_stats = spectral_features(S, fs)
		return np.concatenate([mfcc_vals, spec_stats, [frame_energy]])

	return np.concatenate([mfcc_vals, [frame_energy]])


def get_mfcc_signal(speechSignal: np.ndarray, fs: int, frame_size: float, frame_stride: float,
		num_mel_filters: int, num_ceps: int, use_spectral_features: bool,
		use_mel_filter_bank: bool) -> np.ndarray:

	frame_length = int(frame_size * fs)
	frame_step = int(frame_stride * fs)

	# Frame the signal
	frames = librosa.util.frame(speechSignal, frame_length=frame_length, hop_length=frame_step).T

	mfcc_features = []
	for frame in frames:
		feat = mfcc(frame, fs, num_mel_filters, num_ceps, use_spectral_features, use_mel_filter_bank)
		mfcc_features.append(feat)

	return np.array(mfcc_features)


def save_mfcc(label: str, mfcc_features: np.ndarray, index: int, path: str) -> None:
	label_dir = os.path.join(path, label)
	if not os.path.exists(label_dir):
		os.makedirs(label_dir)
	np.save(f"{label_dir}/mfcc_{index:03d}.npy", mfcc_features)


def create_mfcc_dataset(path: str, file_path: str, include_noisy: bool,
		use_spectral_features: bool, use_mel_filter_bank: bool) -> None:

	print("Creating MFCC dataset...")
	audio_data, labels = file_reader.load_dataset(file_path, include_noisy=include_noisy)
	print("Number of audio files loaded:", len(audio_data))

	for i, sig in enumerate(audio_data):
		label = labels[i]

		signal = preprocess_signal(sig)

		mfcc_raw = get_mfcc_signal(
			signal, fs=16000,
			frame_size=0.025, frame_stride=0.01,
			num_mel_filters=40, num_ceps=13,
			use_spectral_features=use_spectral_features,
			use_mel_filter_bank=use_mel_filter_bank
		)

		mfcc_norm = (mfcc_raw - np.mean(mfcc_raw)) / np.std(mfcc_raw)

		delta = compute_deltas(mfcc_norm)
		delta_delta = compute_deltas(delta)

		final = np.concatenate((mfcc_norm, delta, delta_delta), axis=1)
		save_mfcc(label, final, i, path)

	print("MFCC dataset created.")


def get_mfcc_dataset(path: str, file_path: str, include_noisy: bool,
		use_spectral_features: bool, use_mel_filter_bank: bool) -> tuple[np.ndarray, np.ndarray]:

	if os.listdir(path) == []:
		create_mfcc_dataset(path, file_path, include_noisy, use_spectral_features, use_mel_filter_bank)

	data = []
	labels = []

	for label in os.listdir(path):
		label_dir = os.path.join(path, label)
		if not os.path.isdir(label_dir):
			continue

		for mfcc_file in os.listdir(label_dir):
			if not mfcc_file.endswith('.npy'):
				continue
			mfcc_data = np.load(os.path.join(label_dir, mfcc_file))
			data.append(mfcc_data)
			labels.append(label)

	max_len = max(m.shape[0] for m in data)
	num_features = data[0].shape[1]

	padded = pad_mfcc_features(data, max_len, num_features)
	padded = np.expand_dims(padded, -1)

	return padded, np.array(labels)

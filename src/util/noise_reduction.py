import numpy as np
import soundfile as sf
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa

NOISE_PATH = "data/audio/noise/"
SNR_AMOUNTS = [20]


def load_all_noises() -> list[np.ndarray]:
    noise_files = [f for f in os.listdir(NOISE_PATH) if f.endswith('.wav')]
    noises = []
    for file in noise_files:
        noise_path = os.path.join(NOISE_PATH, file)
        noises.append(sf.read(noise_path)[0])
    return noises


def load_noise(path, target_len):
    noise, _ = sf.read(path)

    # Convert stereo → mono
    if len(noise.shape) > 1:
        noise = np.mean(noise, axis=1)

    noise = noise.astype(np.float32)

    # If shorter → repeat until long enough
    if len(noise) < target_len:
        repeat = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeat)

    # If longer → trim to exact size
    return noise[:target_len]


def preprocess_noise(noise: np.ndarray, target_len: int) -> np.ndarray:
    noise = noise.flatten()
    # Loop/trim to match length
    if len(noise) < target_len:
        repeat = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeat)
    return noise[:target_len]


def show_spectrogram(audio, fs, title):
    f, t, Sxx = signal.spectrogram(audio, fs)
    plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def add_noise(signal, noise, snr_db):
    # Compute signal power
    sig_power = np.mean(signal ** 2)
    
    # Compute noise power for target SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear

    # Scale noise to match required power
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-10)
    noise = noise * np.sqrt(noise_power)

    # Add noise
    return signal + noise

def spectral_subtraction(signal, fs=16000, frame_size=0.025, frame_stride=0.01,
		noise_frames=6, alpha=4.0, beta=0.01):

	# Frame + STFT parameters
	n_fft = int(frame_size * fs)
	hop_length = int(frame_stride * fs)
	window = 'hamming'

	# STFT
	S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window)
	mag, phase = np.abs(S), np.angle(S)

	# Noise estimate from first N frames
	noise_mag = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)

	# Spectral subtraction
	clean_mag = mag - alpha * noise_mag
	clean_mag = np.maximum(clean_mag, beta * noise_mag)

	# Reconstruct complex spectrum
	clean_S = clean_mag * np.exp(1j * phase)

	# Inverse STFT
	cleaned = librosa.istft(clean_S, hop_length=hop_length, window=window)

	return cleaned.astype(np.float32)

def apply_noise_to_signals(signals: list[np.ndarray], labels: np.ndarray, snr_amounts: list = SNR_AMOUNTS) -> tuple[list[np.ndarray], list[str]]:
    noises = load_all_noises()
    noisy_signals = []
    noisy_labels = []
    
    for i, signal in enumerate(signals):
        for noise in noises:
            for snr_db in snr_amounts:
                processed_noise = preprocess_noise(noise, len(signal))
                noisy_signal = add_noise(signal, processed_noise, snr_db)
                noisy_signals.append(noisy_signal)
                noisy_labels.append(labels[i])
    return noisy_signals, noisy_labels


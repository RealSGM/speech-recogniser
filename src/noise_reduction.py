import numpy as np
import soundfile as sf
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


NOISE_PATH = "data/noise/"
SNR_AMOUNTS = [0, 10, 20]


def load_all_noises() -> np.ndarray:
    noise_files = [f for f in os.listdir(NOISE_PATH) if f.endswith('.wav')]
    noises = []
    for file in noise_files:
        noise_path = os.path.join(NOISE_PATH, file)
        noises.append(sf.read(noise_path)[0])
    return np.array(noises)


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


def spectral_subtraction(
        signal, 
        fs=16000, 
        frame_size=0.025, 
        frame_stride=0.01,
        noise_frames=6,
        alpha=6.0,
        beta=0.001  # noise floor factor
    ):
    
    frame_len = int(frame_size * fs)
    frame_step = int(frame_stride * fs)
    window = np.hamming(frame_len)

    # Frame the signal
    frames = []
    for i in range(0, len(signal) - frame_len, frame_step):
        frames.append(signal[i:i+frame_len] * window)
    frames = np.array(frames)

    # FFT
    spectra = np.fft.rfft(frames, axis=1)
    magnitudes = np.abs(spectra)
    phases = np.angle(spectra)

    # Noise magnitude estimate
    noise_mag = np.mean(magnitudes[:noise_frames], axis=0)

    # Spectral subtraction
    clean_mag = magnitudes - alpha * noise_mag
    clean_mag = np.maximum(clean_mag, beta * noise_mag)   # ✅ noise floor instead of 0

    # Reconstruct spectrum
    clean_spec = clean_mag * np.exp(1j * phases)

    # iFFT
    clean_frames = np.fft.irfft(clean_spec, axis=1)

    # Overlap-add (with normalization)
    output_len = (len(clean_frames)-1) * frame_step + frame_len
    cleaned = np.zeros(output_len)
    window_sum = np.zeros(output_len)

    for i in range(len(clean_frames)):
        start = i * frame_step
        cleaned[start:start+frame_len] += clean_frames[i] * window
        window_sum[start:start+frame_len] += window**2

    # Avoid divide-by-zero
    cleaned = cleaned / (window_sum + 1e-10)

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


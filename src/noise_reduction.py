import numpy as np
import soundfile as sf
import os
from feature_extractor import *
import numpy as np


NOISE_PATH = "data/noise/"
SNR_AMOUNTS = [-5, 5, 10, 20]


def load_all_noises() -> np.ndarray:
    noise_files = [f for f in os.listdir(NOISE_PATH) if f.endswith('.wav')]
    noises = []
    for file in noise_files:
        noise_path = os.path.join(NOISE_PATH, file)
        noises.append(sf.read(noise_path)[0])
    return np.array(noises)


def preprocess_noise(noise: np.ndarray, target_len: int) -> np.ndarray:
    noise = noise.flatten()
    # Loop/trim to match length
    if len(noise) < target_len:
        repeat = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeat)
    return noise[:target_len]


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
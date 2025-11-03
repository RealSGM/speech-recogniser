import numpy as np
from scipy.fftpack import dct
import os
import file_reader

save_path = "data/mfccs"


def compute_deltas(mfcc_features: np.ndarray, N: int = 2) -> np.ndarray:
    deltas = np.zeros_like(mfcc_features)
    for t in range(len(mfcc_features)):
        numerator = 0
        denominator = 0
        for n in range(1, N + 1):
            prev = max(0, t - n)
            next = min(len(mfcc_features) - 1, t + n)
            numerator += n * (mfcc_features[next] - mfcc_features[prev])
            denominator += 2 * (n ** 2)
        deltas[t] = numerator / denominator
    return deltas

def pre_emphasise(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def preprocess_signal(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    # Pre-emphasise
    emphasized_signal = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    # Mean normalisation
    normalized_signal = emphasized_signal - np.mean(emphasized_signal)
    return normalized_signal


def standardise_signal_lengths(signals: np.ndarray) -> np.ndarray:
    target_length = max(signal.shape[0] for signal in signals)
    standardized = []
    for signal in signals:
        if signal.shape[0] < target_length:
            pad_width = target_length - signal.shape[0]
            padded_signal = np.pad(signal, ((0, pad_width), (0, 0)), mode='constant')
            standardized.append(padded_signal)
        else:
            truncated_signal = signal[:target_length, :]
            standardized.append(truncated_signal)
    return np.array(standardized)


def mag_and_phase(speechFrame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Apply a Hamming window
    window = np.hamming(len(speechFrame))
    windowed = speechFrame * window

    # Compute FFT
    spectrum = np.fft.fft(windowed)

    # Extract magnitude and phase
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    return magnitude, phase


def hz_to_mel(hz: float) -> float:
    return 1125 * np.log(1 + hz / 700)


def mel_to_hz(mel: float) -> float:
    return 700 * (np.exp(mel / 1125) - 1)


def mel_triangular_filterbank(magspec: np.ndarray, num_channels: int, fs: int) -> np.ndarray:
    num_freqs = len(magspec)
    fbank = np.zeros((num_channels))
    
    # Compute mel-spaced filter points
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(fs / 2)
    mel_points = np.linspace(mel_min, mel_max, num_channels + 2)
    hz_points = np.vectorize(mel_to_hz)(mel_points)
    bin_points = np.floor((num_freqs + 1) * hz_points / fs).astype(int)
    
    for i in range(1, num_channels + 1):
        start = bin_points[i - 1]
        center = bin_points[i]
        end = bin_points[i + 1]
        
        for j in range(start, center):
            fbank[i - 1] += (j - start) / (center - start) * magspec[j]
        for j in range(center, end):
            fbank[i - 1] += (end - j) / (end - center) * magspec[j]
        
    return fbank


def mfcc(speechFrame: np.ndarray, fs: int, num_mel_filters: int, num_ceps: int) -> np.ndarray:
    magnitude, _ = mag_and_phase(speechFrame)
    
    frame_energy = np.log(np.sum(magnitude ** 2) + 1e-10) # Add 1e-10 to avoid log(0)
    
    mel_fbank = mel_triangular_filterbank(magnitude, num_mel_filters, fs)
    log_mel_fbank = np.log(mel_fbank + 1e-10) # - Add 1e-10 to avoid log(0)
    mfcc = dct(log_mel_fbank, type=2, norm='ortho')[:num_ceps]
    return np.append(mfcc, frame_energy)


def get_mfcc_signal(speechSignal: np.ndarray, fs: int, frame_size: float, frame_stride: float, num_mel_filters: int, num_ceps: int) -> np.ndarray:
    signal_length = len(speechSignal)
    frame_length = int(frame_size * fs)
    frame_step = int(frame_stride * fs)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(speechSignal, z)

    mfcc_features = []
    for i in range(num_frames):
        start_index = i * frame_step
        end_index = start_index + frame_length
        frame = pad_signal[start_index:end_index]
        mfcc_feat = mfcc(frame, fs, num_mel_filters, num_ceps)
        mfcc_features.append(mfcc_feat)

    return np.array(mfcc_features)


def save_mfcc(label: str, mfcc_features: np.ndarray, index: int) -> None:
    label_dir = os.path.join(save_path, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    np.save(f"{label_dir}/mfcc_{index:03d}.npy", mfcc_features)


def create_mfcc_dataset() -> None:
    audio_data, labels = file_reader.load_dataset()
    print("Number of audio files loaded:", len(audio_data))
    
    for i in range(len(audio_data)):
        label = labels[i]
        
        signal = preprocess_signal(audio_data[i])
        mfcc = get_mfcc_signal(signal, fs=16000, frame_size=0.025, frame_stride=0.01, num_mel_filters=40, num_ceps=13)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalise MFCC
        delta = compute_deltas(mfcc)
        delta_delta = compute_deltas(delta)
        final = np.concatenate((mfcc, delta, delta_delta), axis=1)
        
        print(f"MFCC+Energy shape: {mfcc.shape}")
        print(f"Final combined feature shape: {final.shape}")
        print(f"MFCC shape for sample {i}: {final.shape} of label '{label}'")
        
        save_mfcc(label, final, i)


def get_mfcc_dataset() -> tuple[np.ndarray, np.ndarray]:
    if os.listdir(save_path) == []:
        create_mfcc_dataset()

    data = []
    labels = []

    for label in os.listdir(save_path):
        label_dir = os.path.join(save_path, label)
        if not os.path.isdir(label_dir):
            continue

        for mfcc_file in os.listdir(label_dir):
            if not mfcc_file.endswith('.npy'):
                continue

            mfcc_data = np.load(os.path.join(label_dir, mfcc_file))
            data.append(mfcc_data)
            labels.append(label)

    # Normalise MFCCs
    data = [mfcc / np.max(np.abs(mfcc)) for mfcc in data]

    # Pad them
    max_len = max(m.shape[0] for m in data)
    num_features = data[0].shape[1]

    padded = np.zeros((len(data), max_len, num_features))
    for i, m in enumerate(data):
        length = m.shape[0]
        padded[i, :length, :] = m

    padded = np.expand_dims(padded, -1)

    return padded, np.array(labels)

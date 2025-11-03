import numpy as np
import os
import file_reader
from scipy.fftpack import dct

SAVE_PATH = "data/mfccs"
MIN_INT = 1e-10


def spectral_features(magnitude: np.ndarray, fs: int) -> np.ndarray:
    freqs = np.linspace(0, fs / 2, len(magnitude)//2)
    mag = magnitude[:len(freqs)]
    mag /= np.sum(mag) + 1e-10

    # Basic stats
    centroid = np.sum(freqs * mag)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag))
    flatness = np.exp(np.mean(np.log(mag + 1e-10))) / (np.mean(mag) + 1e-10)
    
    # Additional features
    cumsum_mag = np.cumsum(mag)
    rolloff = freqs[np.searchsorted(cumsum_mag, 0.85)]  # 85% roll-off
    skewness = np.sum(((freqs - centroid) ** 3) * mag) / (spread**3 + 1e-10)
    kurtosis = np.sum(((freqs - centroid) ** 4) * mag) / (spread**4 + 1e-10)
    slope = (mag[-1] - mag[0]) / (freqs[-1] - freqs[0] + 1e-10)
    
    return np.array([centroid, spread, flatness, rolloff, skewness, kurtosis, slope])



## Ensure all MFCC feature arrays have the same length by padding with zeros
def pad_mfcc_features(data: list[np.ndarray], max_len: int, num_features: int) -> np.ndarray:
    padded = np.zeros((len(data), max_len, num_features))
    for i, m in enumerate(data):
        length = m.shape[0]
        padded[i, :length, :] = m
    return padded


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


## Preprocess the signal by applying pre-emphasis and normalisation
def preprocess_signal(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    emphasised_signal = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    normalised_signal = emphasised_signal - np.mean(emphasised_signal)
    return normalised_signal


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
    frame_energy = np.log(np.sum(magnitude ** 2) + MIN_INT)
    mel_fbank = mel_triangular_filterbank(magnitude, num_mel_filters, fs)
    log_mel_fbank = np.log(mel_fbank + MIN_INT)
    mfcc = dct(log_mel_fbank, type=2, norm='ortho')[:num_ceps]
    
    spec_stats = spectral_features(magnitude, fs)
    
    spec_stats[0] /= (fs / 2)  # centroid
    spec_stats[1] /= (fs / 2)  # spread
    spec_stats[2] = np.clip(spec_stats[2], 0, 1)  # flatness

    return np.concatenate([mfcc, [frame_energy]])


## Extract MFCC features from the speech signal
def get_mfcc_signal(speechSignal: np.ndarray, fs: int, frame_size: float, frame_stride: float, num_mel_filters: int, num_ceps: int) -> np.ndarray:
    # Calculate frame parameters
    signal_length = len(speechSignal)
    frame_length = int(frame_size * fs)
    frame_step = int(frame_stride * fs)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    # Pad signal to make sure that all frames have equal number of samples
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(speechSignal, z)

    # Extract MFCC features for each frame
    mfcc_features = []
    for i in range(num_frames):
        start_index = i * frame_step
        end_index = start_index + frame_length
        frame = pad_signal[start_index:end_index]
        mfcc_feat = mfcc(frame, fs, num_mel_filters, num_ceps)
        mfcc_features.append(mfcc_feat)

    return np.array(mfcc_features)


## Save MFCC features to the disk
def save_mfcc(label: str, mfcc_features: np.ndarray, index: int) -> None:
    label_dir = os.path.join(SAVE_PATH, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    np.save(f"{label_dir}/mfcc_{index:03d}.npy", mfcc_features)


def create_mfcc_dataset() -> None:
    audio_data, labels = file_reader.load_dataset()
    print("Number of audio files loaded:", len(audio_data))
    
    for i in range(len(audio_data)):
        label = labels[i]
        
        signal = preprocess_signal(audio_data[i])
        mfcc = get_mfcc_signal(signal, fs=16000, frame_size=0.025, frame_stride=0.01, num_mel_filters=40, num_ceps=26)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalise MFCC
        delta = compute_deltas(mfcc)
        delta_delta = compute_deltas(delta)
        final = np.concatenate((mfcc, delta, delta_delta), axis=1)
        print(f"MFCC shape for sample {i}: {final.shape} of label '{label}'")
        
        save_mfcc(label, final, i)


## Load MFCC dataset from disk, creating it if necessary
def get_mfcc_dataset() -> tuple[np.ndarray, np.ndarray]:
    if os.listdir(SAVE_PATH) == []:
        create_mfcc_dataset()

    data = []
    labels = []

    for label in os.listdir(SAVE_PATH):
        label_dir = os.path.join(SAVE_PATH, label)
        if not os.path.isdir(label_dir):
            continue

        for mfcc_file in os.listdir(label_dir):
            if not mfcc_file.endswith('.npy'):
                continue

            mfcc_data = np.load(os.path.join(label_dir, mfcc_file))
            data.append(mfcc_data)
            labels.append(label)

    # Pad them
    max_len = max(m.shape[0] for m in data)
    num_features = data[0].shape[1]
    padded = pad_mfcc_features(data, max_len, num_features)
    padded = np.expand_dims(padded, -1)  # Add channel dimension

    return padded, np.array(labels)

if __name__ == "__main__":
    create_mfcc_dataset()
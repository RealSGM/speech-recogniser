import numpy as np
from scipy.fftpack import dct
import os
import file_reader

save_path = "data/mfccs"

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


def linear_rectangular_filterbank(magspec: np.ndarray, num_channels: int):
    num_freqs = len(magspec)
    fbank = np.zeros((num_channels))
    
    for i in range(num_channels):
        r = np.zeros(num_freqs)
        start = int(i * num_freqs / num_channels)
        end = int((i + 1) * num_freqs / num_channels)
        r[start:end] = 1
        fbank[i] = np.matmul(r, magspec)
        
    return fbank


def linear_triangular_filterbank(magspec: np.ndarray, num_channels: int):
    num_freqs = len(magspec)
    fbank = np.zeros((num_channels))
    
    for i in range(num_channels):
        r = np.zeros(num_freqs)
        start = int(i * num_freqs / num_channels)
        end = int((i + 1) * num_freqs / num_channels)
        r[start:end] = 1
        if i > 0:
            r[start:end] *= np.linspace(0, 1, end - start)
        if i < num_channels - 1:
            r[start:end] *= np.linspace(1, 0, end - start)
        fbank[i] = np.matmul(r, magspec)
        
    return fbank


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
    mel_fbank = mel_triangular_filterbank(magnitude, num_mel_filters, fs)
    log_mel_fbank = np.log(mel_fbank + 1e-10) # - Add 1e-10 to avoid log(0)
    mfcc = dct(log_mel_fbank, type=2, norm='ortho')[:num_ceps]
    return mfcc


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


def create_mfcc_dataset() -> None:
    audio_data, labels = file_reader.load_dataset()
    print("Number of audio files loaded:", len(audio_data))
    
    for i in range(len(audio_data)):
        mfcc = get_mfcc_signal(audio_data[i], fs=16000, frame_size=0.025, frame_stride=0.01, num_mel_filters=40, num_ceps=13)
        label = labels[i]
        print(f"MFCC shape for sample {i}: {mfcc.shape} of label '{label}'")
        
        # Check if save directory + label subdirectory exists, create if not
        label_dir = os.path.join(save_path, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Save MFCC features
        np.save(f"{label_dir}/mfcc_{i:03d}.npy", mfcc)


def get_mfcc_dictionary() -> dict:
    if os.listdir(save_path) == []:
        create_mfcc_dataset()

    mfcc_dict = {}

    for label in os.listdir(save_path):
        label_dir = os.path.join(save_path, label)
        if not os.path.isdir(label_dir):
            continue
        
        mfcc_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        mfcc_list = []
        for mfcc_file in mfcc_files:
            mfcc_path = os.path.join(label_dir, mfcc_file)
            mfcc_data = np.load(mfcc_path)
            mfcc_list.append(mfcc_data)
        
        mfcc_dict[label] = mfcc_list
        print(f"Loaded {len(mfcc_list)} MFCC files for label '{label}'")
    
    return mfcc_dict
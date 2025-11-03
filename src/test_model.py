from keras.models import load_model
import pickle
from time import sleep
from feature_extractor import *
import sounddevice as sd
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from time import sleep

from feature_extractor import (
    preprocess_signal,
    get_mfcc_signal,
    compute_deltas
)

# def trim_silence(signal, fs, thresh=0.015):
#     abs_sig = np.abs(signal)
#     mask = abs_sig > thresh
#     if not np.any(mask):
#         return signal  # nothing detected, return raw
#     start = np.argmax(mask)
#     end = len(mask) - np.argmax(mask[::-1])
#     return signal[start:end]

def pad_to_model_length(features, target_len):
    """Pads or trims MFCC to match model input length."""
    current_len = features.shape[0]
    num_features = features.shape[1]

    if current_len > target_len:
        return features[:target_len, :]  # trim

    padded = np.zeros((target_len, num_features))
    padded[:current_len, :] = features
    return padded


def live_predict(model, encoder, fs=16000, duration=2.0, top_n=3):
    print("\nPress ENTER to record...")
    input()
    sleep(0.75)
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    audio = audio.flatten()

    # print("✅ Recorded. Trimming silence...")
    # audio = trim_silence(audio, fs)

    print("✅ Preprocessing...")
    audio = preprocess_signal(audio)

    # === Extract features (must match training) ===
    mfcc = get_mfcc_signal(
        audio, fs=fs,
        frame_size=0.025,
        frame_stride=0.01,
        num_mel_filters=40,
        num_ceps=26
    )

    # Normalise
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)

    delta = compute_deltas(mfcc)
    delta_delta = compute_deltas(delta)

    final = np.concatenate((mfcc, delta, delta_delta), axis=1)

    # === Pad to model length ===
    target_len = model.input_shape[1]      
    num_features = model.input_shape[2]   

    final = pad_to_model_length(final, target_len)
    final = np.expand_dims(final, axis=-1) 
    final = np.expand_dims(final, axis=0) 

    print("Running model prediction...")
    probs = model.predict(final, verbose=0)[0]

    # Get top predictions
    top_idx = np.argsort(probs)[::-1][:top_n]
    classes = encoder.inverse_transform(top_idx)

    print("\n**Prediction results:**")
    for i in range(top_n):
        print(f"{i+1}. {classes[i]} — {probs[top_idx[i]]*100:.2f}%")

    print("\nDone.\n")
    return classes[0]


if __name__ == "__main__":
    # Load model
    model = load_model("data/networks/speech_model.keras")

    # Load label encoder
    with open("data/networks/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    while True:
        live_predict(model, encoder)
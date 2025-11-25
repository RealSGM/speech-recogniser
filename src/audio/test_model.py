from time import sleep
from feature_extractor import *
import sounddevice as sd
import soundfile as sf
import numpy as np
import pickle
from keras.models import load_model
from feature_extractor import preprocess_signal, get_mfcc_signal, compute_deltas
from noise_reduction import load_noise, add_noise, spectral_subtraction, show_spectrogram
import numpy as _np
import os
from collections import Counter


def pad_to_model_length(features, target_len):
    """Pads or trims MFCC to match model input length."""
    current_len = features.shape[0]
    num_features = features.shape[1]

    if current_len > target_len:
        return features[:target_len, :]  # trim

    padded = np.zeros((target_len, num_features))
    padded[:current_len, :] = features
    return padded


def predict(audio, model, encoder, fs=16000, top_n=3, print_results=True):
    audio = audio.flatten()
    audio = preprocess_signal(audio)

    mfcc = get_mfcc_signal(
        audio, fs=fs,
        frame_size=0.025,
        frame_stride=0.01,
        num_mel_filters=40,
        num_ceps=13,
        use_spectral_features=True,
        use_mel_filter_bank=True
    )

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)

    delta = compute_deltas(mfcc)
    delta_delta = compute_deltas(delta)
    final = np.concatenate((mfcc, delta, delta_delta), axis=1)


    target_len = model.input_shape[1]
    final = pad_to_model_length(final, target_len)
    final = np.expand_dims(final, axis=-1) 
    final = np.expand_dims(final, axis=0)   
    probs = model.predict(final, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_n]
    classes = encoder.inverse_transform(top_idx)

    if print_results:
        print("\nPrediction results:")
        for i in range(top_n):
            print(f"{i+1}. {classes[i]} — {probs[top_idx[i]]*100:.2f}%")

    return classes[0]

def live_record(fs=16000, duration=2.0):
    print("\nPress ENTER to record...")
    input()
    sleep(0.75)
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    return audio, fs


def predict_files(files, model_path="data/networks/no_noise/speech_model.keras", encoder_path="data/networks/no_noise/label_encoder.pkl", fs=16000, top_n=1):
    model = load_model(model_path)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    results = []
    for fp in files:
        audio, sr = sf.read(fp)
        audio = audio.flatten()

        if sr != fs:
            target_len = int(len(audio) * (fs / sr))
            audio = _np.interp(_np.linspace(0, len(audio) - 1, target_len), _np.arange(len(audio)), audio)

        pred = predict(audio, model, encoder, fs=fs, top_n=top_n,print_results=False)
        results.append(pred)

    return results

def get_file_paths():
    paths = []
    while True:
        s = input("Enter name and file number (e.g. charlie 2), or press ENTER to finish: ").strip()
        if not s:
            break
        try:
            name, num_str = s.rsplit(" ", 1)
            file_number = int(num_str)
        except ValueError:
            print("Invalid input. Use format: Name Number (e.g. charlie 2)")
            continue
        file_number_str = f"{file_number:03d}"
        path = f"data/testing_recordings/{name}/{name}_{file_number_str}.wav"
        paths.append(path)
    return paths
    
def evaluate_predictions(paths, predictions, print_results=True):
    total = len(paths)
    correct = 0
    confusion = Counter()
    details = []

    for path, pred in zip(paths, predictions):
        expected = os.path.basename(os.path.dirname(path))
        pred_str = str(pred)
        is_correct = pred_str == expected
        correct += int(is_correct)
        confusion[(expected, pred_str)] += 1
        details.append((path, expected, pred_str, is_correct))

    if print_results:
        for path, expected, pred_str, is_correct in details:
            status = "CORRECT" if is_correct else "INCORRECT"
        accuracy = (correct / total * 100) if total else 0.0
        print(f"\nSummary: {correct}/{total} correct — Accuracy: {accuracy:.2f}%")
        print("\n(expected -> predicted):")
        for (exp, pr), cnt in sorted(confusion.items()):
            print(f"  {exp} -> {pr}: {cnt}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "confusion": dict(confusion),
        "details": details,
    }

if __name__ == "__main__":

    ## FILE RECOGNISER

    # paths = get_file_paths()
    # results = predict_files(paths)
    # evaluate_predictions(paths, results)


    ## LIVE DEMO

    model_path = "data/networks/all_features/speech_model.keras"
    encoder_path = "data/networks/all_features/label_encoder.pkl"
    fs = 16000

    model = load_model(model_path)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    try:
        while True:
            audio, sr = live_record(fs=fs, duration=2.0)
            prediction = predict(audio, model, encoder, fs=fs, top_n=3, print_results=True)
            print(f"Top prediction: {prediction}\n")
    except KeyboardInterrupt:
        print("\nExiting live prediction.")

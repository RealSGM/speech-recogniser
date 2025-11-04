from time import sleep
from feature_extractor import *
import sounddevice as sd
import soundfile as sf
import numpy as np
import pickle
from keras.models import load_model
from feature_extractor import preprocess_signal, get_mfcc_signal, compute_deltas
from noise_reduction import load_noise, add_noise, spectral_subtraction, show_spectrogram


def pad_to_model_length(features, target_len):
    """Pads or trims MFCC to match model input length."""
    current_len = features.shape[0]
    num_features = features.shape[1]

    if current_len > target_len:
        return features[:target_len, :]  # trim

    padded = np.zeros((target_len, num_features))
    padded[:current_len, :] = features
    return padded


def predict(audio, model, encoder, fs=16000, top_n=3):
    audio = audio.flatten()
    audio = preprocess_signal(audio)

    mfcc = get_mfcc_signal(
        audio, fs=fs,
        frame_size=0.025,
        frame_stride=0.01,
        num_mel_filters=40,
        num_ceps=26
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

    print("\n**Prediction results:**")
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

if __name__ == "__main__":

    name = "mateusz"

    #name = input("Name: ")
    #audio, fs = live_record()
    audio, fs = sf.read("data/testing_recordings/"+name+"/"+name+"_009.wav")
    audio = audio.flatten()

    # === Load babble noise ===
    noise = load_noise("data/noise/babble.wav", len(audio))
    noisy_audio = add_noise(audio, noise, snr_db=0)
    #cleaned_audio = spectral_subtraction(noisy_audio,fs)
    show_spectrogram(noisy_audio,fs,"Mateusz Test File with 'babble' noise")
    sd.play(noisy_audio,fs)
    sd.wait()
    #model = load_model("data/networks/speech_model.keras")
    # Load label encoder
    # with open("data/networks/label_encoder.pkl", "rb") as f:
    #     encoder = pickle.load(f)
    # predict(noisy_audio,model,encoder)
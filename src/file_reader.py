import os
import soundfile as sf
import numpy as np

RECORDING_DIRECTORY = "data/recordings"


def read_audiofile(file_path: str) -> tuple[np.ndarray, int]:
    data, samplerate = sf.read(file_path)
    return data, samplerate


def load_dataset():
    print("- Loading dataset...")
    audio_data, labels = [], []
    
    for label in os.listdir(RECORDING_DIRECTORY):
        folder = os.path.join(RECORDING_DIRECTORY, label)
        
        if not os.path.isdir(folder):
            continue
    
        for filename in os.listdir(folder):
            if not filename.endswith(".wav"):
                continue
                
            file_path = os.path.join(folder, filename)
            data, _ = read_audiofile(file_path)
            
            audio_data.append(data)
            labels.append(label)
    
    print("- Dataset loaded")
    return audio_data, labels
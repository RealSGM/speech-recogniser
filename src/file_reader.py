import os
import soundfile as sf
import numpy as np

TRAINING_DIRECTORY = "data/training_recordings"
TESTING_DIRECTORY = "data/testing_recordings"


def read_audiofile(file_path: str) -> tuple[np.ndarray, int]:
    data, samplerate = sf.read(file_path)
    return data, samplerate


def load_dataset(path: str) -> tuple[list[np.ndarray], list[str]]:
    print("- Loading dataset...")
    audio_data, labels = [], []
    
    for label in os.listdir(path):
        folder = os.path.join(path, label)
        
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
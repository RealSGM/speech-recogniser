import os
import soundfile as sf
import numpy as np

recording_directory = "data/recordings"


def read_audiofile(file_path: str) -> tuple[np.ndarray, int]:
    data, samplerate = sf.read(file_path)
    return data, samplerate


def load_dataset():
    audio_data, labels = [], []
    
    for label in os.listdir(recording_directory):
        print("-" * 20)
        print(f"Processing label: {label}")
        folder = os.path.join(recording_directory, label)
        
        if not os.path.isdir(folder):
            continue
    
        for filename in os.listdir(folder):
            if not filename.endswith(".wav"):
                continue
                
            file_path = os.path.join(folder, filename)
            data, _ = read_audiofile(file_path)
            
            audio_data.append(data)
            labels.append(label)
            
    return audio_data, labels

if __name__ == "__main__":
    load_dataset()
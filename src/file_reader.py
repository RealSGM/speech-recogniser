import os
import soundfile as sf
import numpy as np
import noise_reduction as nr

TRAINING_DIRECTORY = "data/training_recordings"
TESTING_DIRECTORY = "data/testing_recordings"


def read_audiofile(file_path: str) -> tuple[np.ndarray, int]:
    data, samplerate = sf.read(file_path)
    return data, samplerate


def load_dataset(path: str, include_noisy: bool = False) -> tuple[list[np.ndarray], list[str]]:
    audio_data = []
    labels = []
    
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
    
    if include_noisy:
        new_audio_data, new_labels = nr.apply_noise_to_signals(audio_data, np.array(labels))
        audio_data.extend(new_audio_data)
        labels.extend(new_labels)
    
    return audio_data, labels
import file_reader
from feature_extractor import get_mfcc_signal
import numpy as np
import os

save_path = "data/mfccs"

def create_mfcc_dataset() -> None:
    audio_data, labels = file_reader.load_dataset()
    print("Number of audio files loaded:", len(audio_data))
    
    for i in range(len(audio_data)):
        mfcc = get_mfcc_signal(audio_data[i], fs=16000, frame_size=0.025, frame_stride=0.01, num_mel_filters=40, num_ceps=13)
        label = labels[i]
        print(f"MFCC shape for sample {i}: {mfcc.shape}")
        
        # Check if save directory + label subdirectory exists, create if not
        label_dir = os.path.join(save_path, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Save MFCC features
        np.save(f"{label_dir}/mfcc_{i:03d}.npy", mfcc)


if __name__ == "__main__":
    create_mfcc_dataset()
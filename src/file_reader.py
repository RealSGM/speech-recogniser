import os
import numpy as np
import cv2
import moviepy as mp

TRAINING_PATH = "data/recordings_training"
TESTING_PATH = "data/recordings_testing"


def get_data_from_mp4(file_path: str) -> tuple[np.ndarray, float, np.ndarray, int]:
    cap = cv2.VideoCapture(file_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    video_data = np.array(frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    clip = mp.VideoFileClip(file_path)
    audio = clip.audio
    audio_sample_rate = audio.fps # type: ignore
    audio_data = audio.to_soundarray() # type: ignore
    
    return video_data, fps, audio_data, audio_sample_rate


def load_dataset(recordings_path: str) -> tuple[list[np.ndarray], list[str], list[float], list[np.ndarray], list[int]]:
    
    # Check if the path exists
    if not os.path.exists(recordings_path):
        os.makedirs(recordings_path)
        
    
    labels = []
    video_data = []
    video_fps = []
    audio_data_array = []
    audio_sample_rates = []
    
    for label in os.listdir(recordings_path):
        folder = os.path.join(recordings_path, label)
        
        if not os.path.isdir(folder):
            continue
        
        for filename in os.listdir(folder):
            if not filename.endswith(".mp4"):
                continue
            
            file_path = os.path.join(folder, filename)
            video, fps, audio_data, audio_sample_rate = get_data_from_mp4(file_path)
            
            video_data.append(video)
            labels.append(label)
            video_fps.append(fps)
            audio_data_array.append(audio_data)
            audio_sample_rates.append(audio_sample_rate)
    
    return video_data, labels, video_fps, audio_data_array, audio_sample_rates

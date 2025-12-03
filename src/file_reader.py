import os
import numpy as np
import cv2
import moviepy as mp
import feature_extractor.audio_extractor as afe
import feature_extractor.visual_extractor as vfe
from utils.config import MainConfig


PATH = "data/"
RECORDINGS_PATH = "recordings/"
TESTING = "testing/"
TRAINING = "training/"
VECTORS = "vectors/"
AUDIO = "audio/"
VISUAL = "visual/"


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


def get_wav_data(path: str) -> tuple[list[np.ndarray], list[str], list[float], list[np.ndarray], list[int]]:
    labels = []
    video_data = []
    video_fps = []
    audio_data_array = []
    audio_sample_rates = []
    
    print(f"Loading data from path: {path}")
    
    for label in os.listdir(path):
        folder = os.path.join(path, label)
        
        if not os.path.isdir(folder):
            continue
        
        print(f"Processing folder: {folder}")
        
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
            # break ## REMOVEME
    
    return video_data, labels, video_fps, audio_data_array, audio_sample_rates


def load_dataset(type: str, config: MainConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    print(f"Loading dataset of type: {type}")
    full_recording_path = f"{PATH}{RECORDINGS_PATH}{type}/"
    full_mfcc_path = f"{PATH}{VECTORS}{AUDIO}{type}/"
    full_visual_path = f"{PATH}{VECTORS}{VISUAL}{type}/"
    
    # Check if the path exists
    if not os.path.exists(full_recording_path):
        print(f"Recording path {full_recording_path} does not exist. Creating directory.")
        os.makedirs(full_recording_path)

    if not os.path.exists(full_mfcc_path):
        print(f"MFCC path {full_mfcc_path} does not exist. Creating directory.")
        os.makedirs(full_mfcc_path)
        
    if not os.path.exists(full_visual_path):
        print(f"Visual path {full_visual_path} does not exist. Creating directory.")
        os.makedirs(full_visual_path)
        
    labels = []
    
    ## Check if MFCCs exist, if not create them
    if os.listdir(full_mfcc_path) == [] or os.listdir(full_visual_path) == []:
        video_data, labels, video_fps, audio_data_array, _ = get_wav_data(full_recording_path)
        afe.create_dataset(audio_data_array, labels, path=full_mfcc_path, config=config)
        vfe.create_dataset(video_data, labels, video_fps, path=full_visual_path, config=config)  
    
    mfcc_data, _ = afe.get_dataset(full_mfcc_path)
    visual_data, _ = vfe.get_dataset(full_visual_path)
    
    return mfcc_data, visual_data, labels
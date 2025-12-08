import os
import shutil
import json
import numpy as np
import cv2
import moviepy as mp
import feature_extractor.audio_extractor as afe
import feature_extractor.visual_extractor as vfe
from utils.config import MainConfig

PATH = "data/"
CONFIGS = "configs/"
RECORDINGS_PATH = "recordings/"
TESTING = "testing/"
TRAINING = "training/"
VECTORS = "vectors/"
AUDIO = "audio/"
VISUAL = "visual/"

def create_config_path(config: MainConfig) -> str:
    p = get_config_path(config)
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(f"{p}{VECTORS}{AUDIO}{TRAINING}", exist_ok=True)
    os.makedirs(f"{p}{VECTORS}{AUDIO}{TESTING}", exist_ok=True)
    os.makedirs(f"{p}{VECTORS}{VISUAL}{TRAINING}", exist_ok=True)
    os.makedirs(f"{p}{VECTORS}{VISUAL}{TESTING}", exist_ok=True)
    with open(f"{p}/results.json","w") as f: f.write("{}")
    return p

def get_config_path(config: MainConfig) -> str:
    return f"{PATH}{CONFIGS}{config.test_name}/"

def get_results_dict(config: MainConfig) -> dict:
    fpath = f"{get_config_path(config)}/results.json"
    try:
        with open(fpath,"r") as f: return json.load(f)
    except: return {}

def add_to_results_file(config: MainConfig, results: dict):
    fpath = f"{get_config_path(config)}/results.json"
    d = get_results_dict(config)
    d.update(results)
    with open(fpath,"w") as f: json.dump(d,f,indent=4)

def get_data_from_mp4(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0: fps = 25.0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    video = np.array(frames, dtype=np.uint8)

    clip = mp.VideoFileClip(path)
    audio_data = clip.audio.to_soundarray()
    audio_sr = clip.audio.fps
    clip.close()

    return video, fps, audio_data, audio_sr

def load_dataset(type: str, config: MainConfig):
    root = f"{PATH}{RECORDINGS_PATH}{type}/"
    mfcc_path = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{AUDIO}{type}/"
    vis_path = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{VISUAL}{type}/"

    os.makedirs(mfcc_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)

    labels = []

    if len(os.listdir(mfcc_path)) == 0 or len(os.listdir(vis_path)) == 0:
        for label in sorted(os.listdir(root)):
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                continue

            videos = []
            audios = []
            fps_list = []
            labs = []

            for fname in sorted(os.listdir(label_dir)):
                if fname.endswith(".mp4"):
                    p = os.path.join(label_dir, fname)
                    v, fps, a, sr = get_data_from_mp4(p)
                    videos.append(v)
                    fps_list.append(fps)
                    audios.append(a)
                    labs.append(label)
                    labels.append(label)

            afe.create_dataset(audios, labs, path=mfcc_path, config=config)
            vfe.create_dataset(videos, labs, fps_list, path=vis_path, config=config)

            del videos, audios, fps_list, labs
            import gc; gc.collect()

    mfcc, _ = afe.get_dataset(mfcc_path)
    vis, _ = vfe.get_dataset(vis_path)
    return mfcc, vis, labels

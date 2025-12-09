import os
import shutil
import json
import numpy as np
import cv2
import moviepy as mp
import feature_extractor.audio_extractor as afe
import feature_extractor.visual_extractor as vfe
import gc
from utils.config import MainConfig
from concurrent.futures import ProcessPoolExecutor

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
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    video = np.array(frames, dtype=np.uint8)

    clip = mp.VideoFileClip(path)
    audio_data = clip.audio.to_soundarray() # type: ignore
    audio_sr = clip.audio.fps # type: ignore
    clip.close()

    return video, fps, audio_data, audio_sr


def get_audio_visual_recordings(label: str, root: str) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[str]]:
	label_dir = os.path.join(root, label)
	videos = []
	audios = []
	fps_list = []
	labs = []

	files = [f for f in sorted(os.listdir(label_dir)) if f.endswith(".mp4")]
	total = len(files)

	for i, fname in enumerate(files):
		print(f"\r    Loading {label} {i + 1}/{total:>2}", end="", flush=True)

		p = os.path.join(label_dir, fname)
		v, fps, a, sr = get_data_from_mp4(p)

		videos.append(v)
		fps_list.append(fps)
		audios.append(a)
		labs.append(label)
	print()

	return videos, audios, fps_list, labs


def get_labels_from_files(type: str) -> list[str]:
    root = f"{PATH}{RECORDINGS_PATH}{type}/"
    labels = []

    for label in sorted(os.listdir(root)):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue

        files = [f for f in sorted(os.listdir(label_dir)) if f.endswith(".mp4")]
        labels.extend([label] * len(files))

    return labels


def load_dataset(type: str, config: MainConfig):
    root = f"{PATH}{RECORDINGS_PATH}{type}/"
    mfcc_path = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{AUDIO}{type}/"
    vis_path = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{VISUAL}{type}/"

    os.makedirs(mfcc_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)

    if len(os.listdir(mfcc_path)) == 0 or len(os.listdir(vis_path)) == 0:
        for label in sorted(os.listdir(root)):
            label_dir = os.path.join(root, label)
            
            if not os.path.isdir(label_dir):
                continue
            
            print(f"\n\033[36mProcessing label: {label}\033[0m")
            videos, audios, fps_list, labs = get_audio_visual_recordings(label=label, root=root)
            print(f"\033[33mExtracting audio features for {len(audios)} recordings...\033[0m")
            afe.create_dataset(audios, labs, path=mfcc_path, cfg=config)
            print(f"\033[33mExtracting video features for {len(videos)} recordings...\033[0m")
            vfe.create_dataset(videos, labs, fps_list, path=vis_path, cfg=config)

            del videos, audios, fps_list, labs
            import gc; gc.collect()

    labels = get_labels_from_files(type)
    mfcc, _ = afe.get_dataset(mfcc_path)
    vis, _ = vfe.get_dataset(vis_path)
    return mfcc, vis, labels


# Load one set of audio and video data
# Use that data to generate all the MFCC and visual vectors for all configs
def create_all_datasets(config_files: list[MainConfig]) -> None:
    testing_root = f"{PATH}{RECORDINGS_PATH}{TESTING}/"
    training_root = f"{PATH}{RECORDINGS_PATH}{TRAINING}/"
    
    for label in sorted(os.listdir(testing_root)):
        label_dir = os.path.join(testing_root, label)
        if not os.path.isdir(label_dir):
            continue
        
        # Clear terminal
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"\n==============================")
        print(f" PROCESSING LABEL: {label}")
        print(f"==============================\n")
        
        testing_video, testing_audio, testing_fps, testing_labels = get_audio_visual_recordings(label=label, root=testing_root)
        training_video, training_audio, training_fps, training_labels = get_audio_visual_recordings(label=label, root=training_root)
        
        for config in config_files:
            print(f"\n\033[36mProcessing config: {config.test_name}, label: {label}\033[0m")
            
            mfcc_path_test = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{AUDIO}{TESTING}/"
            vis_path_test = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{VISUAL}{TESTING}/"
            mfcc_path_train = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{AUDIO}{TRAINING}/"
            vis_path_train = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{VISUAL}{TRAINING}/"
            
            create_config_path(config)
            
            print(f"\033[33mExtracting audio features for {len(testing_audio)} testing recordings...\033[0m")
            afe.create_dataset(testing_audio, testing_labels, path=mfcc_path_test, cfg=config)
            print(f"\033[33mExtracting video features for {len(testing_video)} testing recordings...\033[0m")
            vfe.create_dataset(testing_video, testing_labels, testing_fps, path=vis_path_test, cfg=config)
            
            print(f"\033[33mExtracting audio features for {len(training_audio)} training recordings...\033[0m")
            afe.create_dataset(training_audio, training_labels, path=mfcc_path_train, cfg=config)
            print(f"\033[33mExtracting video features for {len(training_video)} training recordings...\033[0m")
            vfe.create_dataset(training_video, training_labels, training_fps, path=vis_path_train, cfg=config)
        
        del testing_video, testing_audio, testing_fps, testing_labels
        del training_video, training_audio, training_fps, training_labels
        gc.collect()


def process_label_config(args):
    label, config, testing_root, training_root = args

    print(f"\033[36mProcessing config: {config.test_name}, label: {label}\033[0m")

    testing_video, testing_audio, testing_fps, testing_labels = get_audio_visual_recordings(
        label=label, root=testing_root
    )

    training_video, training_audio, training_fps, training_labels = get_audio_visual_recordings(
        label=label, root=training_root
    )

    mfcc_path_test = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{AUDIO}{TESTING}/"
    vis_path_test  = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{VISUAL}{TESTING}/"
    mfcc_path_train = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{AUDIO}{TRAINING}/"
    vis_path_train  = f"{PATH}{CONFIGS}{config.test_name}/{VECTORS}{VISUAL}{TRAINING}/"

    create_config_path(config)

    afe.create_dataset(testing_audio, testing_labels, path=mfcc_path_test, cfg=config)
    vfe.create_dataset(testing_video, testing_labels, testing_fps, path=vis_path_test, cfg=config)

    afe.create_dataset(training_audio, training_labels, path=mfcc_path_train, cfg=config)
    vfe.create_dataset(training_video, training_labels, training_fps, path=vis_path_train, cfg=config)

    del testing_video, testing_audio, testing_fps, testing_labels
    del training_video, training_audio, training_fps, training_labels
    gc.collect()
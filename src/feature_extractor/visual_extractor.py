import os
import numpy as np
import cv2
import mediapipe as mp
from skimage.feature import local_binary_pattern
from utils.config import MainConfig

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False) # type: ignore

last_valid_roi = None
last_gray = None
last_height = None

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = 16


def save_vector_file(label: str, features: np.ndarray, index: int, path: str) -> None:
	label_dir = os.path.join(path, label)
	os.makedirs(label_dir, exist_ok=True)
	np.save(f"{label_dir}/vector_{index:03d}.npy", features)


def extract_mouth_roi(frame: np.ndarray, cfg: MainConfig) -> tuple[np.ndarray, list] | None:
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	result = mp_face.process(rgb)

	if not result.multi_face_landmarks:
		return None

	h, w, _ = frame.shape
	points = []

	for i in cfg.visual.lip_landmarks:
		lm = result.multi_face_landmarks[0].landmark[i]
		points.append((int(lm.x*w), int(lm.y*h)))

	x, y, ww, hh = cv2.boundingRect(np.array(points))

	p = cfg.visual.pad_px
	x = max(0, x-p)
	y = max(0, y-p)
	ww = min(w-x, ww+2*p)
	hh = min(h-y, hh+2*p)

	roi = frame[y:y+hh, x:x+ww]
	return roi, points


def reset_visual_state():
	global last_gray, last_valid_roi, last_height
	last_gray = None
	last_valid_roi = None
	last_height = None


def extract_lbp_features(gray):
	lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method='uniform')
	hist,_ = np.histogram(lbp, bins=LBP_BINS, range=(0,LBP_BINS), density=True)
	return hist.astype(np.float32)


def extract_frame_features(frame: np.ndarray, cfg: MainConfig) -> np.ndarray:
	global last_valid_roi, last_gray, last_height

	roi_data = extract_mouth_roi(frame, cfg)
	if roi_data is None:
		if last_valid_roi is None:
			dim = 2 * cfg.visual.coeff_size**2
			if cfg.visual.use_motion_stats:
				dim += 2
			if cfg.visual.use_lbp:
				dim += LBP_BINS
			return np.zeros(dim, dtype=np.float32)
		roi = last_valid_roi.copy()
		pts = None
	else:
		roi, pts = roi_data
		last_valid_roi = roi.copy()

	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, cfg.visual.roi_size).astype(np.float32)
	gray /= 255.0

	if last_gray is None:
		flow = np.zeros_like(gray)
	else:
		flow = cv2.absdiff(gray, last_gray)

	last_gray = gray.copy()

	dct_gray = cv2.dct(gray)
	dct_flow = cv2.dct(flow)

	k = cfg.visual.coeff_size
	dct_gray = dct_gray[:k, :k]
	dct_flow = dct_flow[:k, :k]

	features = [
		dct_gray.flatten(),
		dct_flow.flatten()
	]

	if cfg.visual.use_motion_stats:
		mean_motion = np.mean(flow)
		std_motion = np.std(flow)
		features.append(np.array([mean_motion, std_motion], dtype=np.float32))

	if cfg.visual.use_lbp:
		lbp = extract_lbp_features(gray)
		features.append(lbp)

	return np.concatenate(features)


def compute_deltas(features, N=2):
	padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
	deltas = []

	for t in range(N, len(padded) - N):
		num = 0
		den = 0
		for i in range(1, N + 1):
			num += i * (padded[t+i] - padded[t-i])
			den += 2 * i*i
		deltas.append(num / den)

	return np.array(deltas)


def pad_visual(seqs):
	max_len = max(x.shape[0] for x in seqs)
	num_feat = seqs[0].shape[1]

	padded = np.zeros((len(seqs), max_len, num_feat), dtype=np.float32)

	for i, s in enumerate(seqs):
		L = min(s.shape[0], max_len)
		padded[i, :L, :] = s[:L]

	return padded


def get_dataset(visual_path):
	data = []
	labels = []

	for label in os.listdir(visual_path):
		d = os.path.join(visual_path, label)
		if not os.path.isdir(d):
			continue

		for f in os.listdir(d):
			if f.endswith(".npy"):
				data.append(np.load(os.path.join(d,f)))
				labels.append(label)

	if not data:
		return np.array([]), np.array([])

	X = pad_visual(data)
	X = np.expand_dims(X, -1)

	return X, np.array(labels)


def create_dataset(videos, labels, fps_list, path, cfg: MainConfig) -> None:
	total = len(videos)

	for i, video in enumerate(videos):
		print(f"\r    Processing video {i + 1}/{total:>2}", end="", flush=True)
		reset_visual_state()

		label = labels[i]
		features = []

		for frame in video[::cfg.visual.frame_stride]:
			features.append(extract_frame_features(frame, cfg))

		features = np.array(features)

		if cfg.visual.use_deltas:
			d = compute_deltas(features)
			dd = compute_deltas(d)
			features = np.concatenate([features, d, dd], axis=1)

		save_vector_file(label, features, i, path)

	print()

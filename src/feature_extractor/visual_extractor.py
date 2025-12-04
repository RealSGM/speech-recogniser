import os
import numpy as np
import cv2
import mediapipe as mp


mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False) # type: ignore

LIP_IDS = [
	61,146,91,181,84,17,314,405,321,375,
	291,308,324,318,402,317,14,87,178,88
]


def save_vector_file(label: str, mfcc_features: np.ndarray, index: int, path: str) -> None:
	label_dir = os.path.join(path, label)
	if not os.path.exists(label_dir):
		os.makedirs(label_dir)
	np.save(f"{label_dir}/vector_{index:03d}.npy", mfcc_features)

def extract_mouth_roi(frame):
    # Convert from BGR to RGB
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Use Face Mesh to detect facial landmarks
	result = mp_face.process(rgb)

	if not result.multi_face_landmarks:
		return None

	h, w, _ = frame.shape
	pts = []

	# Get lip landmarks
	for i in LIP_IDS:
		lm = result.multi_face_landmarks[0].landmark[i]
		pts.append((int(lm.x * w), int(lm.y * h)))

	# Create bounding box around lips
	x, y, w, h = cv2.boundingRect(np.array(pts))
	roi = frame[y:y+h, x:x+w]

	return roi


def extract_frame_features(frame):
	roi = extract_mouth_roi(frame)

	if roi is None or roi.size == 0:
		return np.zeros(64)

	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # Grayscale
	gray = cv2.resize(gray, (96, 96)) # Resize to 96x96
	gray = gray.astype(np.float32) / 255.0 # Normalise

	# Perform DCT
	dct = cv2.dct(gray)
	# Take top-left 8x8 coefficients
	coeffs = dct[:8, :8]

	# Flatten to 64-dim vector
	return coeffs.flatten()


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


def pad_visual(data):
	max_len = max(x.shape[0] for x in data)
	num_feat = data[0].shape[1]

	padded = np.zeros((len(data), max_len, num_feat))
	for i, seq in enumerate(data):
		padded[i, :seq.shape[0], :] = seq

	return padded


def get_dataset(visual_path):
	data = []
	labels = []

	for label in os.listdir(visual_path):
		label_dir = os.path.join(visual_path, label)
		if not os.path.isdir(label_dir):
			continue

		for f in os.listdir(label_dir):
			if f.endswith(".npy"):
				data.append(np.load(os.path.join(label_dir, f)))
				labels.append(label)

	if not data:
		return np.array([]), np.array([])

	padded = pad_visual(data)
	padded = np.expand_dims(padded, -1)

	return padded, np.array(labels)


def create_dataset(video_data, labels, video_fps, path, config) -> None:
	print("Creating Visual Feature Dataset...")

	for i, video in enumerate(video_data):
		label = labels[i]
		print(f"Processing video {i+1}/{len(video_data)} - Label: {label}")

		features = []

		for frame in video:
			features.append(extract_frame_features(frame))
		features = np.array(features)

		# Compute deltas and delta-deltas
		delta = compute_deltas(features)
		delta_delta = compute_deltas(delta)

		final = np.concatenate((features, delta, delta_delta), axis=1)

		save_vector_file(label, final, i, path)


import cv2
import mediapipe as mp
import numpy as np

import mediapipe.python.solutions.face_mesh as mp_face

RESIZE_VECTOR = (96, 96)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False, 
    refine_landmarks=True,
    max_num_faces=1
)


def extract_mouth(frame: np.ndarray) -> np.ndarray | None:
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks: # type: ignore
        return None

    lm = result.multi_face_landmarks[0] # type: ignore

    mouth_idx = list(range(61, 88))
    xs, ys = [], []

    for i in mouth_idx:
        x = int(lm.landmark[i].x * w)
        y = int(lm.landmark[i].y * h)
        xs.append(x)
        ys.append(y)

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    if x2 <= x1 or y2 <= y1:
        return frame  # return the original frame if invalid bounding box

    return frame[y1:y2, x1:x2]



def process_frame(frame: np.ndarray) -> np.ndarray:
    # Greyscale conversion
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize frames
    resized_frame = cv2.resize(processed_frame, RESIZE_VECTOR)
    # Normalise between [0, 1]
    normalised_frame = resized_frame / 255.0
    
    return normalised_frame


def extract_dct_features(frames: list[np.ndarray], num_coefficients: int = 20) -> np.ndarray:
    dct_features = []
    
    for frame in frames:
        dct_frame = cv2.dct(np.float32(frame)) # type: ignore
        dct_frame = dct_frame[:num_coefficients, :num_coefficients]
        dct_features.append(dct_frame.flatten())
    
    return np.array(dct_features)


def extract_visual_features(visual_frames: list[np.ndarray]) -> np.ndarray:
    cropped_frames = [extract_mouth(frame) for frame in visual_frames]
    processed_frames = [process_frame(frame) for frame in cropped_frames]     # type: ignore
    dct_features = extract_dct_features(processed_frames)
    return dct_features
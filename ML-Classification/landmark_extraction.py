import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIRS = [
    "asl_alphabet_train/asl_alphabet_train",
    "ASL-HG American Sign Language Hand Gesture Image D/ASL_HG_36000/asl_dataset",
    "my_dataset",  # custom images captured with collect_dataset.py
]
EXCLUDE_CLASSES = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
SEQUENCE_LEN = 20          # frames per sample (static images are replicated to this length)
OUTPUT_FILE  = "asl_landmarks_sequences.npz"
LANDMARKER_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype="float32")
    coords -= coords[0]
    scale = np.max(np.abs(coords))
    if scale < 1e-6:
        return None
    coords /= scale
    return coords  # shape [21, 3]


def detect_landmarks(img_path):
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        return None
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    return normalize_landmarks(result.hand_landmarks[0])


def is_sequence_class(label_path):
    """True if the class folder contains subdirectories (sequence clips)."""
    return any(
        os.path.isdir(os.path.join(label_path, f))
        for f in os.listdir(label_path)
    )


def extract_static(label_path, label, X_list, y_list):
    img_files = [f for f in os.listdir(label_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for img_name in img_files:
        lm = detect_landmarks(os.path.join(label_path, img_name))
        if lm is None:
            continue
        # Replicate the single frame to fill the sequence window
        seq = np.stack([lm] * SEQUENCE_LEN, axis=0)  # [SEQUENCE_LEN, 21, 3]
        X_list.append(seq.reshape(SEQUENCE_LEN, 63))
        y_list.append(label)


def extract_sequences(label_path, label, X_list, y_list):
    seq_dirs = sorted(
        d for d in os.listdir(label_path)
        if os.path.isdir(os.path.join(label_path, d))
    )
    for seq_dir in seq_dirs:
        seq_path = os.path.join(label_path, seq_dir)
        frame_files = sorted(
            f for f in os.listdir(seq_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(frame_files) < SEQUENCE_LEN:
            continue  # skip incomplete sequences

        frames = []
        last_good = None
        for frame_file in frame_files[:SEQUENCE_LEN]:
            lm = detect_landmarks(os.path.join(seq_path, frame_file))
            if lm is None:
                if last_good is not None:
                    frames.append(last_good)  # hold last good frame
                else:
                    frames = []
                    break
            else:
                last_good = lm
                frames.append(lm)

        if len(frames) == SEQUENCE_LEN:
            seq = np.stack(frames, axis=0).reshape(SEQUENCE_LEN, 63)
            X_list.append(seq)
            y_list.append(label)


X_list, y_list = [], []

for data_dir in DATA_DIRS:
    if not os.path.isdir(data_dir):
        print(f"Skipping (not found): {data_dir}")
        continue

    for label in sorted(os.listdir(data_dir)):
        if label in EXCLUDE_CLASSES:
            continue

        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue

        print(f"  [{os.path.basename(data_dir)}] {label}", end=" ... ", flush=True)

        before = len(X_list)
        if is_sequence_class(label_path):
            extract_sequences(label_path, label, X_list, y_list)
        else:
            extract_static(label_path, label, X_list, y_list)

        print(f"{len(X_list) - before} samples")

X = np.array(X_list, dtype="float32")  # [N, SEQUENCE_LEN, 63]
y = np.array(y_list)

np.savez_compressed(OUTPUT_FILE, X=X, y=y)

print(f"\nExtraction complete. {len(X)} samples → {OUTPUT_FILE}")
print(f"Input shape: {X.shape}")

unique, counts = np.unique(y, return_counts=True)
print("\nPer-class counts:")
for cls, cnt in sorted(zip(unique, counts)):
    print(f"  {cls:>8}: {cnt}")

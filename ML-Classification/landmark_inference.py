import json
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

MODEL_PATH       = "asl_landmark_inference_model.keras"
LANDMARKER_PATH  = "hand_landmarker.task"
CLASS_NAMES_PATH = "class_names.json"
SEQUENCE_LEN     = 20
CONFIDENCE_THRESHOLD = 0.6

with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

model = load_model(MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# Ring buffer: holds the last SEQUENCE_LEN normalized landmark frames
frame_buffer = deque(maxlen=SEQUENCE_LEN)


def normalize_landmarks(hand_landmarks):
    """Must match landmark_extraction.py exactly."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype="float32")
    coords -= coords[0]
    scale = np.max(np.abs(coords))
    if scale < 1e-6:
        return None
    coords /= scale
    return coords.reshape(63)  # flat [63]


cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

last_label  = "?"
last_conf   = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        lm_flat = normalize_landmarks(hand_landmarks)

        if lm_flat is not None:
            frame_buffer.append(lm_flat)

            if len(frame_buffer) == SEQUENCE_LEN:
                seq = np.array(frame_buffer, dtype="float32").reshape(1, SEQUENCE_LEN, 63)
                preds = model.predict(seq, verbose=0)
                conf  = float(np.max(preds))
                idx   = int(np.argmax(preds))

                if conf >= CONFIDENCE_THRESHOLD:
                    last_label = CLASS_NAMES[idx]
                    last_conf  = conf
                    color = (0, 255, 0)
                else:
                    last_label = "?"
                    last_conf  = conf
                    color = (0, 165, 255)
            else:
                color = (200, 200, 200)

        # Bounding box from landmarks
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        x_min = max(0, int(min(x_coords) * w) - 20)
        y_min = max(0, int(min(y_coords) * h) - 20)
        x_max = min(w, int(max(x_coords) * w) + 20)
        y_max = min(h, int(max(y_coords) * h) + 20)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, f"{last_label} ({last_conf:.2f})",
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Buffer fill indicator (useful for dynamic signs — shows when model is "ready")
        fill = len(frame_buffer) / SEQUENCE_LEN
        cv2.rectangle(frame, (10, h - 20), (10 + int(200 * fill), h - 5), (0, 200, 255), -1)
        cv2.rectangle(frame, (10, h - 20), (210, h - 5), (255, 255, 255), 1)
    else:
        # No hand: clear buffer so stale frames don't pollute dynamic sign detection
        frame_buffer.clear()

    cv2.imshow("ASL Landmark Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

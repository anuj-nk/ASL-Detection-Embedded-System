import json
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# ── Tuning ────────────────────────────────────────────────────────────────────
DWELL_FRAMES         = 15    # frames candidate must hold before commit (~1.5s @ 30fps)
COOLDOWN_FRAMES      = 10    # frames to lock out after a commit
WINDOW_SIZE          = 5    # sliding window of predictions
MAJORITY_THRESHOLD   = 0.75  # fraction of window that must agree
CONFIDENCE_THRESHOLD = 0.70  # per-frame model confidence floor
SEQUENCE_LEN         = 20

# These tuned values work well with how trash my model is rn. As the model is better it can be faster

MODEL_PATH       = "asl_landmark_inference_model.keras"
LANDMARKER_PATH  = "hand_landmarker.task"
CLASS_NAMES_PATH = "class_names.json"

# ── Setup ─────────────────────────────────────────────────────────────────────
with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

model = load_model(MODEL_PATH)

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
)

frame_buffer   = deque(maxlen=SEQUENCE_LEN)
pred_window    = deque(maxlen=WINDOW_SIZE)
word           = []
dwell_count    = 0
cooldown       = 0
candidate      = None
del_streak     = 0


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype="float32")
    coords -= coords[0]
    scale = np.max(np.abs(coords))
    if scale < 1e-6:
        return None
    coords /= scale
    return coords.reshape(63)


cap = cv2.VideoCapture(0)
print("ASL Word Writer  |  'c' = clear  |  'q' = quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    pred_label = None
    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        lm_flat = normalize_landmarks(lms)
        if lm_flat is not None:
            frame_buffer.append(lm_flat)
            if len(frame_buffer) == SEQUENCE_LEN:
                seq   = np.array(frame_buffer, dtype="float32").reshape(1, SEQUENCE_LEN, 63)
                preds = model.predict(seq, verbose=0)
                conf  = float(np.max(preds))
                if conf >= CONFIDENCE_THRESHOLD:
                    pred_label = CLASS_NAMES[int(np.argmax(preds))]

                x_coords = [lm.x for lm in lms]
                y_coords = [lm.y for lm in lms]
                x1 = max(0, int(min(x_coords) * w) - 20)
                y1 = max(0, int(min(y_coords) * h) - 20)
                x2 = min(w, int(max(x_coords) * w) + 20)
                y2 = min(h, int(max(y_coords) * h) + 20)
                color = (0, 255, 0) if pred_label else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{pred_label or '?'} ({conf:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        frame_buffer.clear()

    # ── Commit logic ──────────────────────────────────────────────────────────
    pred_window.append(pred_label)

    if cooldown > 0:
        cooldown -= 1
    else:
        non_none = [l for l in pred_window if l is not None]
        if non_none:
            top = max(set(non_none), key=non_none.count)
            if non_none.count(top) / WINDOW_SIZE >= MAJORITY_THRESHOLD:
                if top == candidate:
                    dwell_count += 1
                else:
                    candidate, dwell_count = top, 1

                if dwell_count >= DWELL_FRAMES:
                    if top == "del":
                        del_streak += 1
                        to_remove = 1 if del_streak <= 2 else min(2 ** (del_streak - 2), len(word))
                        for _ in range(to_remove):
                            if word: word.pop()
                    elif top == "space":
                        del_streak = 0
                        word.append(" ")
                    else:
                        del_streak = 0
                        word.append(top)
                    print(f'[COMMIT] {top}  ->  "{"".join(word)}"')
                    cooldown, dwell_count, candidate = COOLDOWN_FRAMES, 0, None
            else:
                candidate, dwell_count = None, 0
        else:
            candidate, dwell_count = None, 0

    # ── Overlay ───────────────────────────────────────────────────────────────
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - 55), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f'Word: {"".join(word)}', (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    progress = min(dwell_count / DWELL_FRAMES, 1.0) if DWELL_FRAMES else 0.0
    cv2.rectangle(frame, (10, h - 72), (210, h - 60), (255, 255, 255), 1)
    if cooldown:
        cv2.rectangle(frame, (10, h - 72), (210, h - 60), (60, 60, 60), -1)
    elif progress > 0:
        cv2.rectangle(frame, (10, h - 72), (10 + int(200 * progress), h - 60), (0, 200, 255), -1)
    if candidate:
        cv2.putText(frame, f"-> {candidate}", (220, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    cv2.imshow("ASL Word Writer", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        word.clear()
        print("[CLEAR]")

cap.release()
cv2.destroyAllWindows()
print(f'\nFinal: "{"".join(word)}"')

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "best_model_phase2.h5"
LANDMARKER_PATH = 'hand_landmarker.task' # Ensure this file is in your folder
IMG_SIZE = 96

CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','space','delete','nothing'
]

# -----------------------------
# LOAD MODELS
# -----------------------------
# Load your Keras Classifier
model = load_model(MODEL_PATH)

# Setup MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# START CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks
    detection_result = detector.detect(mp_image)

    # Check if any hands were detected
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            
            # Get bounding box from landmarks
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]

            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            # Add padding and ensure within frame boundaries
            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                input_img = preprocess(hand_img)

                # Prediction
                preds = model.predict(input_img, verbose=0)
                class_idx = np.argmax(preds)
                confidence = np.max(preds)

                label = f"{CLASS_NAMES[class_idx]} ({confidence:.2f})"

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

    cv2.imshow("ASL Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
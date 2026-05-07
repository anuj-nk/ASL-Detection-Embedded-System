import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Edit this list to capture only the signs you need.
CLASSES = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z","del","space",
]
# These signs require motion sequences instead of single images.
SEQUENCE_CLASSES = {"J", "Z"}
IMAGES_PER_CLASS = 200   # static signs: number of images
SEQS_PER_CLASS   = 200   # dynamic signs: number of sequences
SEQUENCE_LEN     = 20    # frames per sequence
CROP_SIZE        = 128   # saved image size in pixels (square); keeps storage tiny
CROP_PAD         = 30    # pixels of padding around the hand bounding box
OUTPUT_DIR       = "my_dataset"
LANDMARKER_PATH  = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)


def draw_landmarks(display, hand_landmarks, w, h):
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)


def crop_hand(frame, hand_landmarks):
    """Crop to the hand bounding box, padded and resized to CROP_SIZE×CROP_SIZE."""
    h, w, _ = frame.shape
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    x1 = max(0, int(min(xs) * w) - CROP_PAD)
    y1 = max(0, int(min(ys) * h) - CROP_PAD)
    x2 = min(w, int(max(xs) * w) + CROP_PAD)
    y2 = min(h, int(max(ys) * h) + CROP_PAD)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


print("Controls: SPACE = capture | N = skip class | Q = quit")

for class_name in CLASSES:
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    is_seq = class_name in SEQUENCE_CLASSES
    target = SEQS_PER_CLASS if is_seq else IMAGES_PER_CLASS

    if is_seq:
        existing = len([d for d in os.listdir(class_dir)
                        if os.path.isdir(os.path.join(class_dir, d))])
    else:
        existing = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])

    count = existing
    mode_label = "SEQUENCE" if is_seq else "IMAGE"
    print(f"\n── {class_name} [{mode_label}]  ({target - count} remaining) ──")
    if is_seq:
        print(f"   Press SPACE when ready — will auto-record {SEQUENCE_LEN} frames")

    skip = False
    while count < target:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        hand_detected = bool(result.hand_landmarks)
        if hand_detected:
            draw_landmarks(display, result.hand_landmarks[0], w, h)

        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(display, f"Sign: {class_name}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(display, f"{count}/{target} {mode_label}S", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        hint = "SPACE=record seq  N=next  Q=quit" if is_seq else "SPACE=capture  N=next  Q=quit"
        cv2.putText(display, hint, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Dataset Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Quit.")
            exit()
        elif key == ord("n"):
            print(f"  Skipping {class_name} (captured {count})")
            skip = True
            break
        elif key == ord(" "):
            if is_seq:
                # Auto-record SEQUENCE_LEN frames, saving hand crops only
                seq_dir = os.path.join(class_dir, f"seq_{count:04d}")
                os.makedirs(seq_dir, exist_ok=True)
                frames_saved = 0
                last_crop = None
                while frames_saved < SEQUENCE_LEN:
                    ret2, f2 = cap.read()
                    if not ret2:
                        break
                    f2 = cv2.flip(f2, 1)
                    h2, w2, _ = f2.shape

                    # Detect hand and crop; fall back to last good crop
                    rgb2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                    res2 = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2))
                    if res2.hand_landmarks:
                        last_crop = crop_hand(f2, res2.hand_landmarks[0])
                    to_save = last_crop if last_crop is not None else cv2.resize(f2, (CROP_SIZE, CROP_SIZE))
                    cv2.imwrite(os.path.join(seq_dir, f"frame_{frames_saved:02d}.jpg"), to_save,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

                    prog = f2.copy()
                    bar_w = int(w2 * frames_saved / SEQUENCE_LEN)
                    cv2.rectangle(prog, (0, h2 - 20), (bar_w, h2), (0, 255, 0), -1)
                    cv2.putText(prog, f"Recording {frames_saved+1}/{SEQUENCE_LEN}",
                                (10, h2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Dataset Collector", prog)
                    cv2.waitKey(50)  # ~20 fps capture rate
                    frames_saved += 1

                count += 1
                print(f"  Saved seq_{count-1:04d} ({frames_saved} frames)")
            else:
                if hand_detected:
                    crop = crop_hand(frame, result.hand_landmarks[0])
                    if crop is not None:
                        img_path = os.path.join(class_dir, f"{class_name}_{count:04d}.jpg")
                        cv2.imwrite(img_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        count += 1
                        print(f"  Saved {img_path}")
                else:
                    print("  No hand detected — move closer and try again")

    if not skip:
        print(f"  {class_name} complete.")

cap.release()
cv2.destroyAllWindows()
print("\nCollection done. Run landmark_extraction.py next.")

# ASL Word Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `word_writer.py` — a webcam ASL script that accumulates signed letters into a word using a sliding-window majority vote for robust letter confirmation.

**Architecture:** Two focused helper classes (`WordBuffer`, `CommitDetector`) handle testable logic; the main loop wires them to the existing MediaPipe + Keras inference pipeline from `landmark_inference.py`. All tuning constants live at the top of the file.

**Tech Stack:** Python 3, TensorFlow/Keras, MediaPipe, OpenCV, pytest

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `word_writer.py` | Create | Config constants, `WordBuffer`, `CommitDetector`, `normalize_landmarks`, `draw_overlay`, main webcam loop |
| `tests/test_word_writer.py` | Create | Unit tests for `WordBuffer` and `CommitDetector` |

---

## Task 1: Scaffold `word_writer.py` and implement `WordBuffer`

**Files:**
- Create: `word_writer.py`
- Create: `tests/__init__.py`
- Create: `tests/test_word_writer.py`

- [ ] **Step 1: Create the test file with `WordBuffer` tests**

Create `tests/__init__.py` (empty), then create `tests/test_word_writer.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from word_writer import WordBuffer


def test_add_single_letter():
    wb = WordBuffer()
    wb.add("H")
    assert wb.text == "H"


def test_add_multiple_letters():
    wb = WordBuffer()
    for c in ["H", "E", "L", "L", "O"]:
        wb.add(c)
    assert wb.text == "HELLO"


def test_del_removes_last_letter():
    wb = WordBuffer()
    wb.add("H")
    wb.add("I")
    wb.add("del")
    assert wb.text == "H"


def test_del_on_empty_buffer_is_safe():
    wb = WordBuffer()
    wb.add("del")
    assert wb.text == ""


def test_space_inserts_space_character():
    wb = WordBuffer()
    wb.add("H")
    wb.add("space")
    wb.add("I")
    assert wb.text == "H I"


def test_clear_empties_buffer():
    wb = WordBuffer()
    wb.add("H")
    wb.add("I")
    wb.clear()
    assert wb.text == ""


def test_clear_then_add():
    wb = WordBuffer()
    wb.add("X")
    wb.clear()
    wb.add("A")
    assert wb.text == "A"
```

- [ ] **Step 2: Create a minimal `word_writer.py` stub so the import resolves**

```python
# word_writer.py  (stub — will be completed in Task 3)

DWELL_FRAMES         = 45
COOLDOWN_FRAMES      = 30
WINDOW_SIZE          = 20
MAJORITY_THRESHOLD   = 0.75
CONFIDENCE_THRESHOLD = 0.70
SEQUENCE_LEN         = 20

MODEL_PATH       = "asl_landmark_inference_model.keras"
LANDMARKER_PATH  = "hand_landmarker.task"
CLASS_NAMES_PATH = "class_names.json"


class WordBuffer:
    pass


class CommitDetector:
    pass
```

- [ ] **Step 3: Run the tests — expect failures**

```bash
cd "ML-Classification" && python -m pytest tests/test_word_writer.py -v
```

Expected: several FAILED with `AttributeError: 'WordBuffer' object has no attribute 'text'`

- [ ] **Step 4: Implement `WordBuffer`**

Replace the `WordBuffer` class stub in `word_writer.py`:

```python
class WordBuffer:
    def __init__(self):
        self._buf = []

    def add(self, label: str):
        if label == "del":
            if self._buf:
                self._buf.pop()
        elif label == "space":
            self._buf.append(" ")
        else:
            self._buf.append(label)

    def clear(self):
        self._buf.clear()

    @property
    def text(self) -> str:
        return "".join(self._buf)
```

- [ ] **Step 5: Run the WordBuffer tests — expect all pass**

```bash
python -m pytest tests/test_word_writer.py -v -k "WordBuffer or test_add or test_del or test_space or test_clear"
```

Expected: 7 PASSED

- [ ] **Step 6: Commit**

```bash
git add word_writer.py tests/__init__.py tests/test_word_writer.py
git commit -m "feat: add WordBuffer with tests"
```

---

## Task 2: Implement `CommitDetector`

**Files:**
- Modify: `word_writer.py` — replace `CommitDetector` stub
- Modify: `tests/test_word_writer.py` — add `CommitDetector` tests

- [ ] **Step 1: Add `CommitDetector` tests to `tests/test_word_writer.py`**

Append to the end of `tests/test_word_writer.py`:

```python
from word_writer import CommitDetector


def _det():
    """Small detector for fast tests: window=5, majority=0.6, dwell=3, cooldown=2."""
    return CommitDetector(window_size=5, majority_threshold=0.6,
                          dwell_frames=3, cooldown_frames=2)


def test_no_commit_before_dwell_completes():
    d = _det()
    results = [d.update("A") for _ in range(2)]
    assert all(r is None for r in results)


def test_commit_fires_after_dwell_frames():
    d = _det()
    results = [d.update("A") for _ in range(3)]
    assert results[-1] == "A"


def test_dwell_resets_on_candidate_change():
    d = _det()
    d.update("A")
    d.update("A")
    d.update("B")          # candidate switches → dwell resets
    results = [d.update("B") for _ in range(3)]
    assert results[-1] == "B"


def test_cooldown_prevents_immediate_second_commit():
    d = _det()
    for _ in range(3):
        d.update("A")      # first commit fires
    # now in cooldown (2 frames)
    results = [d.update("A") for _ in range(2)]
    assert all(r is None for r in results)


def test_commit_possible_after_cooldown_expires():
    d = _det()
    for _ in range(3):
        d.update("A")      # commit + enter cooldown
    for _ in range(2):
        d.update("A")      # burn cooldown
    results = [d.update("A") for _ in range(3)]
    assert results[-1] == "A"


def test_none_frames_count_against_majority():
    d = _det()
    # window=[None, None, None, "A", "A"] → ratio = 2/5 = 0.4 < 0.6 → no candidate
    for _ in range(3):
        d.update(None)
    d.update("A")
    result = d.update("A")
    assert result is None


def test_dwell_progress_increases():
    d = _det()
    d.update("A")
    assert abs(d.dwell_progress - 1 / 3) < 0.01


def test_dwell_progress_zero_before_candidate():
    d = _det()
    assert d.dwell_progress == 0.0


def test_candidate_property():
    d = _det()
    d.update("A")
    assert d.candidate == "A"


def test_in_cooldown_property():
    d = _det()
    for _ in range(3):
        d.update("A")
    assert d.in_cooldown is True
```

- [ ] **Step 2: Run the new tests — expect failures**

```bash
python -m pytest tests/test_word_writer.py -v -k "CommitDetector or _det or test_no_commit or test_commit or test_dwell or test_cooldown or test_none or test_candidate or test_in_cool"
```

Expected: multiple FAILED with `AttributeError`

- [ ] **Step 3: Implement `CommitDetector`**

Replace the `CommitDetector` stub in `word_writer.py`:

```python
from collections import deque

class CommitDetector:
    def __init__(self, window_size: int, majority_threshold: float,
                 dwell_frames: int, cooldown_frames: int):
        self._window = deque(maxlen=window_size)
        self._window_size = window_size
        self._majority_threshold = majority_threshold
        self._dwell_frames = dwell_frames
        self._cooldown_frames = cooldown_frames
        self._dwell_count = 0
        self._cooldown_counter = 0
        self._candidate = None

    def update(self, label):
        """
        Call once per frame with the predicted label (str) or None if confidence
        was below threshold. Returns the committed label on commit, else None.
        """
        self._window.append(label)

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return None

        non_none = [l for l in self._window if l is not None]
        if not non_none:
            self._candidate = None
            self._dwell_count = 0
            return None

        most_common = max(set(non_none), key=non_none.count)
        ratio = non_none.count(most_common) / len(self._window)

        if ratio >= self._majority_threshold:
            if most_common == self._candidate:
                self._dwell_count += 1
            else:
                self._candidate = most_common
                self._dwell_count = 1

            if self._dwell_count >= self._dwell_frames:
                committed = self._candidate
                self._cooldown_counter = self._cooldown_frames
                self._dwell_count = 0
                self._candidate = None
                return committed
        else:
            self._candidate = None
            self._dwell_count = 0

        return None

    @property
    def dwell_progress(self) -> float:
        """0.0–1.0 fraction toward commit, for the UI progress bar."""
        if self._dwell_frames == 0:
            return 0.0
        return min(self._dwell_count / self._dwell_frames, 1.0)

    @property
    def candidate(self):
        return self._candidate

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown_counter > 0
```

Also add `from collections import deque` at the top of `word_writer.py` if not already there.

- [ ] **Step 4: Run all tests — expect all pass**

```bash
python -m pytest tests/test_word_writer.py -v
```

Expected: 17 PASSED

- [ ] **Step 5: Commit**

```bash
git add word_writer.py tests/test_word_writer.py
git commit -m "feat: add CommitDetector with sliding window and cooldown"
```

---

## Task 3: Complete the main webcam loop in `word_writer.py`

**Files:**
- Modify: `word_writer.py` — add imports, `normalize_landmarks`, `draw_overlay`, and main loop

No new tests needed for the webcam loop (OpenCV/camera code is not unit-testable). Manual verification below.

- [ ] **Step 1: Replace the entire contents of `word_writer.py` with the finished script**

```python
import json
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# ── Tuning ───────────────────────────────────────────────────────────────────
DWELL_FRAMES         = 45    # frames candidate must dominate before commit
COOLDOWN_FRAMES      = 30    # frames to ignore after a commit
WINDOW_SIZE          = 20    # prediction sliding-window size
MAJORITY_THRESHOLD   = 0.75  # fraction of window that must agree
CONFIDENCE_THRESHOLD = 0.70  # per-frame model confidence floor
SEQUENCE_LEN         = 20    # must match training

MODEL_PATH       = "asl_landmark_inference_model.keras"
LANDMARKER_PATH  = "hand_landmarker.task"
CLASS_NAMES_PATH = "class_names.json"


# ── WordBuffer ────────────────────────────────────────────────────────────────
class WordBuffer:
    def __init__(self):
        self._buf = []

    def add(self, label: str):
        if label == "del":
            if self._buf:
                self._buf.pop()
        elif label == "space":
            self._buf.append(" ")
        else:
            self._buf.append(label)

    def clear(self):
        self._buf.clear()

    @property
    def text(self) -> str:
        return "".join(self._buf)


# ── CommitDetector ────────────────────────────────────────────────────────────
class CommitDetector:
    def __init__(self, window_size: int, majority_threshold: float,
                 dwell_frames: int, cooldown_frames: int):
        self._window = deque(maxlen=window_size)
        self._window_size = window_size
        self._majority_threshold = majority_threshold
        self._dwell_frames = dwell_frames
        self._cooldown_frames = cooldown_frames
        self._dwell_count = 0
        self._cooldown_counter = 0
        self._candidate = None

    def update(self, label):
        self._window.append(label)

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return None

        non_none = [l for l in self._window if l is not None]
        if not non_none:
            self._candidate = None
            self._dwell_count = 0
            return None

        most_common = max(set(non_none), key=non_none.count)
        ratio = non_none.count(most_common) / len(self._window)

        if ratio >= self._majority_threshold:
            if most_common == self._candidate:
                self._dwell_count += 1
            else:
                self._candidate = most_common
                self._dwell_count = 1

            if self._dwell_count >= self._dwell_frames:
                committed = self._candidate
                self._cooldown_counter = self._cooldown_frames
                self._dwell_count = 0
                self._candidate = None
                return committed
        else:
            self._candidate = None
            self._dwell_count = 0

        return None

    @property
    def dwell_progress(self) -> float:
        if self._dwell_frames == 0:
            return 0.0
        return min(self._dwell_count / self._dwell_frames, 1.0)

    @property
    def candidate(self):
        return self._candidate

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown_counter > 0


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_landmarks(hand_landmarks):
    """Must match landmark_extraction.py exactly."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype="float32")
    coords -= coords[0]
    scale = np.max(np.abs(coords))
    if scale < 1e-6:
        return None
    coords /= scale
    return coords.reshape(63)


def draw_overlay(frame, word_text: str, dwell_progress: float,
                 candidate, in_cooldown: bool):
    h, w = frame.shape[:2]

    # Semi-transparent word bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"Word: {word_text}", (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Dwell progress bar
    bar_top, bar_bot = h - 72, h - 60
    cv2.rectangle(frame, (10, bar_top), (210, bar_bot), (255, 255, 255), 1)
    if dwell_progress > 0 and not in_cooldown:
        cv2.rectangle(frame, (10, bar_top),
                      (10 + int(200 * dwell_progress), bar_bot), (0, 200, 255), -1)
    elif in_cooldown:
        cv2.rectangle(frame, (10, bar_top), (210, bar_bot), (80, 80, 80), -1)

    # Candidate label next to bar
    if candidate:
        cv2.putText(frame, f"-> {candidate}", (220, bar_bot),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)


# ── Main ──────────────────────────────────────────────────────────────────────
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

frame_buffer  = deque(maxlen=SEQUENCE_LEN)
word_buf      = WordBuffer()
commit_det    = CommitDetector(WINDOW_SIZE, MAJORITY_THRESHOLD,
                               DWELL_FRAMES, COOLDOWN_FRAMES)

cap = cv2.VideoCapture(0)
print("ASL Word Writer  |  'c' clear  |  'q' quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        lm_flat = normalize_landmarks(hand_landmarks)

        pred_label = None
        if lm_flat is not None:
            frame_buffer.append(lm_flat)
            if len(frame_buffer) == SEQUENCE_LEN:
                seq   = np.array(frame_buffer, dtype="float32").reshape(1, SEQUENCE_LEN, 63)
                preds = model.predict(seq, verbose=0)
                conf  = float(np.max(preds))
                idx   = int(np.argmax(preds))
                if conf >= CONFIDENCE_THRESHOLD:
                    pred_label = CLASS_NAMES[idx]

                # Bounding box
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]
                x_min = max(0, int(min(x_coords) * w) - 20)
                y_min = max(0, int(min(y_coords) * h) - 20)
                x_max = min(w, int(max(x_coords) * w) + 20)
                y_max = min(h, int(max(y_coords) * h) + 20)
                color = (0, 255, 0) if pred_label else (0, 165, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                label_txt = f"{pred_label or '?'} ({conf:.2f})"
                cv2.putText(frame, label_txt, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        frame_buffer.clear()

    committed = commit_det.update(pred_label)
    if committed:
        word_buf.add(committed)
        print(f"[COMMIT] {committed}  ->  \"{word_buf.text}\"")

    draw_overlay(frame, word_buf.text, commit_det.dwell_progress,
                 commit_det.candidate, commit_det.in_cooldown)

    cv2.imshow("ASL Word Writer", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        word_buf.clear()
        print("[CLEAR]")

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal word: \"{word_buf.text}\"")
```

- [ ] **Step 2: Run all unit tests — verify nothing broke**

```bash
python -m pytest tests/test_word_writer.py -v
```

Expected: 17 PASSED

- [ ] **Step 3: Smoke-test the script manually**

```bash
python word_writer.py
```

Verify:
- Webcam opens and shows the word bar at the bottom
- Holding a sign for ~1.5 s commits the letter and prints `[COMMIT] X -> "X"` in the terminal
- Signing `del` removes the last character
- Signing `space` inserts a space
- Pressing `c` clears the word; pressing `q` prints the final word and exits
- The dwell progress bar fills as you hold a sign and turns grey during cooldown

- [ ] **Step 4: Commit**

```bash
git add word_writer.py
git commit -m "feat: complete word_writer main loop with overlay and terminal output"
```

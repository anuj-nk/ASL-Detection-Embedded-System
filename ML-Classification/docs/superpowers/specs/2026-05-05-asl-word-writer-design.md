# ASL Word Writer — Design Spec
**Date:** 2026-05-05

## Overview

A new script `word_writer.py` that extends the existing real-time ASL landmark inference to let a user spell out full words, letter by letter, using hold-based sign confirmation.

## Goals

- Let a user sign A–Z continuously and build up a word character by character
- Handle `del` (backspace) and `space` as first-class classes
- Be robust to a noisy/imperfect model via a sliding-window majority vote
- Keep all tuning knobs configurable at the top of the file
- Eventually port to ESP32 (TFLite Micro) — design for simplicity and fixed-size buffers

## Non-Goals

- Embedded port (future work)
- Word prediction / autocomplete
- Multi-hand support

---

## Configuration Block

All tuning constants at the top of `word_writer.py`:

```python
DWELL_FRAMES         = 45    # frames candidate must dominate before commit (~1.5s @ 30fps)
COOLDOWN_FRAMES      = 30    # frames to ignore after a commit (prevents double-letters)
WINDOW_SIZE          = 20    # size of the prediction sliding window
MAJORITY_THRESHOLD   = 0.75  # fraction of window that must agree on the candidate
CONFIDENCE_THRESHOLD = 0.70  # per-frame model confidence floor (below = None in window)
SEQUENCE_LEN         = 20    # must match training config
```

---

## Commit Logic (Sliding Window + Cooldown)

A `deque(maxlen=WINDOW_SIZE)` accumulates per-frame predicted labels on top of the model's output (not its input). The model's existing `frame_buffer` of 20 landmark frames is unchanged.

**Each frame:**
1. If model confidence ≥ `CONFIDENCE_THRESHOLD`, append the predicted label; otherwise append `None`
2. Find the most common non-`None` label in the window
3. If that label occupies ≥ `MAJORITY_THRESHOLD` of the window, it becomes the **candidate**
4. A `dwell_count` increments each frame the candidate stays the same; resets on candidate change
5. When `dwell_count` reaches `DWELL_FRAMES` and `cooldown_counter == 0` → **commit**
6. After commit: reset `dwell_count`, set `cooldown_counter = COOLDOWN_FRAMES`
7. `cooldown_counter` decrements each frame; no commits fire while it is > 0

---

## Word Buffer & Special Characters

`word_buffer` is a plain Python list of strings.

| Committed label | Action |
|---|---|
| A–Z | `word_buffer.append(letter)` |
| `del` | `word_buffer.pop()` if non-empty |
| `space` | `word_buffer.append(" ")` |

Current word: `"".join(word_buffer)`

**Keybindings:**
- `c` — clear the entire word buffer
- `q` — quit

---

## Display

### OpenCV Overlay
- Semi-transparent dark bar at bottom of frame: `Word: HELLO`
- Dwell progress bar (same style as existing buffer-fill indicator) showing how close the current candidate is to committing

### Terminal Output
One `print()` per commit event:
```
[COMMIT] H  →  "H"
[COMMIT] I  →  "HI"
[COMMIT] del  →  "H"
[COMMIT] space  →  "H "
```

---

## File Changes

| File | Change |
|---|---|
| `word_writer.py` | New script, based on `landmark_inference.py` |
| All existing files | Unchanged |

---

## Future: ESP32 Port Notes

- Replace `deque` with fixed C arrays of the same sizes
- Replace MediaPipe with an on-device hand landmark model (e.g., a quantized MobileNet-based detector or a dedicated hand keypoint model for TFLite Micro)
- The commit logic (majority window + cooldown) maps directly to integer counters — no heap allocation needed
- Terminal output maps to UART serial output

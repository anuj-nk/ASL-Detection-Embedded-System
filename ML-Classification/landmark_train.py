import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras import layers, models, callbacks

DATASET_PATH       = "asl_landmarks_sequences.npz"
MODEL_SAVE_PATH    = "asl_landmark_model.keras"
INFERENCE_MODEL_PATH = "asl_landmark_inference_model.keras"
CLASS_NAMES_PATH   = "class_names.json"
SEQUENCE_LEN       = 20
MIN_SAMPLES        = 2

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading dataset...")
data = np.load(DATASET_PATH, allow_pickle=True)
X = data["X"].astype("float32")   # [N, SEQUENCE_LEN, 63]
y_raw = data["y"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)

# Drop classes with too few samples
counts = np.bincount(y_encoded)
valid_mask = counts[y_encoded] >= MIN_SAMPLES
if not valid_mask.all():
    removed = [encoder.classes_[i] for i in np.where(counts < MIN_SAMPLES)[0]]
    print(f"Dropping under-represented classes: {removed}")
    X, y_encoded = X[valid_mask], y_encoded[valid_mask]
    y_str = encoder.inverse_transform(y_encoded)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_str)

num_classes = len(encoder.classes_)
print(f"{num_classes} classes: {list(encoder.classes_)}")
print(f"Dataset shape: {X.shape}")

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(list(encoder.classes_), f)

# ── Mirror augmentation ───────────────────────────────────────────────────────
# Negate x-coordinates across all frames to make model hand-agnostic.
x_cols = list(range(0, 63, 3))  # p0_x, p1_x, ..., p20_x
X_mirror = X.copy()
X_mirror[:, :, x_cols] *= -1
X = np.concatenate([X, X_mirror], axis=0)
y_encoded = np.concatenate([y_encoded, y_encoded], axis=0)
print(f"After mirror augmentation: {len(X)} samples")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}")

# ── Noise augmentation (applied to training set only) ─────────────────────────
noise = np.random.normal(0, 0.005, X_train.shape).astype("float32")
X_train = np.concatenate([X_train, X_train + noise], axis=0)
y_train = np.concatenate([y_train, y_train], axis=0)
print(f"After noise augmentation: {len(X_train)} training samples")

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(SEQUENCE_LEN, 63)),
    layers.BatchNormalization(),
    layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
    layers.Dropout(0.3),
    layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
cbs = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy",
                              save_best_only=True, verbose=1),
]

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    callbacks=cbs,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nVal accuracy: {acc:.4f}  Val loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print(classification_report(y_val, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(
    ax=ax, colorbar=False, xticks_rotation=45)
ax.set_title("Validation Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("Saved training_history.png")

# ── Inference model (no Dropout — required for clean TFLite conversion) ───────
inference_model = models.Sequential([
    layers.Input(shape=(SEQUENCE_LEN, 63)),
    layers.BatchNormalization(),
    layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
    layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])
src = [l for l in model.layers if l.get_weights()]
dst = [l for l in inference_model.layers if l.get_weights()]
for s, d in zip(src, dst):
    d.set_weights(s.get_weights())
inference_model.save(INFERENCE_MODEL_PATH)
print(f"\nInference model saved to {INFERENCE_MODEL_PATH}")
print("Next: run convert_to_tflite.py")

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# Change these paths and settings to match your setup
# ─────────────────────────────────────────────────────────────

# Path to the downloaded Kaggle dataset folder
DATA_DIR = "asl_alphabet_train/asl_alphabet_train"

# Where to save the trained model
OUTPUT_MODEL_PATH = "asl_model.h5"

# Input image size fed into MobileNetV2
# 96x96 is small enough for ESP32 but large enough for accuracy
IMG_SIZE = 96

# Number of images processed per training step
# Reduce to 16 if you run out of RAM
BATCH_SIZE = 32

# Maximum number of training rounds
# EarlyStopping will stop sooner if validation stops improving
MAX_EPOCHS = 30

# How much of the data to use for validation (0.2 = 20%)
VALIDATION_SPLIT = 0.2

# Number of output classes (A-Z with delete, space, nothing = 29 total)
NUM_CLASSES = 29

# How many MobileNetV2 layers to unfreeze for fine tuning
# Set to 0 to skip fine tuning (faster but less accurate)
FINETUNE_LAYERS = 30

# ─────────────────────────────────────────────────────────────
# STEP 1: VERIFY GPU / CPU AVAILABILITY
# ─────────────────────────────────────────────────────────────

print("\n=== Device Info ===")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU available: {gpus}")
    # Allow GPU memory to grow instead of allocating all at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, training on CPU (will be slower)")

print(f"TensorFlow version: {tf.__version__}")


# ─────────────────────────────────────────────────────────────
# STEP 2: DATA LOADING AND AUGMENTATION
#
# ImageDataGenerator handles:
#   - Rescaling pixel values from 0-255 to 0.0-1.0
#   - Splitting into train and validation sets
#   - On-the-fly augmentation to reduce overfitting
#
# NOTE: horizontal_flip=False because ASL signs are NOT symmetric
# Flipping would create incorrect mirror signs
# ─────────────────────────────────────────────────────────────

print("\n=== Loading Dataset ===")

# Augmentation only applied to training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,           # normalize pixels to [0, 1]
    validation_split=VALIDATION_SPLIT,
    rotation_range=10,           # random rotation up to 10 degrees
    width_shift_range=0.1,       # random horizontal shift up to 10%
    height_shift_range=0.1,      # random vertical shift up to 10%
    shear_range=0.1,             # slight shear transformation
    zoom_range=0.1,              # random zoom up to 10%
    brightness_range=[0.8, 1.2], # vary brightness to handle lighting changes
    horizontal_flip=False,       # IMPORTANT: do NOT flip ASL signs
    fill_mode="nearest"          # fill empty pixels after transforms
)

# Validation data only gets rescaled, no augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT
)

# Load training images from the dataset directory
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),  # resize all images to 96x96
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",          # one-hot encoded labels for 26 classes
    shuffle=True
)

# Load validation images from the same directory (different split)
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    shuffle=False                      # no need to shuffle validation data
)

# Save the class index mapping so we can decode predictions later
class_indices = train_generator.class_indices
print(f"\nClass mapping (first 5): {dict(list(class_indices.items())[:5])}")
print(f"Total training samples  : {train_generator.samples}")
print(f"Total validation samples: {val_generator.samples}")

# Save class names to a file for use during inference on ESP32
class_names = list(class_indices.keys())
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")
print("Saved class_names.txt")


# ─────────────────────────────────────────────────────────────
# STEP 3: BUILD THE MODEL
#
# Strategy: Transfer Learning with MobileNetV2
#
# MobileNetV2 was pretrained on ImageNet (1.4M images, 1000 classes)
# We remove its top classification layer and add our own for 26 ASL classes
#
# Phase 1: Train only our new top layers (base frozen)
# Phase 2: Unfreeze last N layers and fine tune at a low learning rate
#
# MobileNetV2 is specifically designed to be small and fast, making it
# well suited for conversion to TFLite int8 for the ESP32
# ─────────────────────────────────────────────────────────────

print("\n=== Building Model ===")

# Load MobileNetV2 without its original top classification layer
# include_top=False means we add our own classifier
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"   # start from ImageNet pretrained weights
)

# Freeze all base model layers for phase 1 training
# We only want to train our new top layers first
base_model.trainable = False

print(f"Base model layers: {len(base_model.layers)}")
print(f"Base model parameters: {base_model.count_params():,}")

# Build the classification head on top of MobileNetV2
x = base_model.output

# Global average pooling reduces spatial dimensions to a single vector
# This is more efficient than Flatten and helps prevent overfitting
x = layers.GlobalAveragePooling2D()(x)

# Dense layer to learn ASL-specific features
x = layers.Dense(256, activation="relu")(x)

# Dropout randomly sets 40% of neurons to zero during training
# This prevents the model from memorizing training data
x = layers.Dropout(0.4)(x)

# Another dense layer for more complex feature combinations
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# Output layer: 26 neurons, one per ASL letter
# Softmax converts raw scores to probabilities that sum to 1
output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

# Combine base model and custom head into one model
model = Model(inputs=base_model.input, outputs=output)

# Compile for phase 1 training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ─────────────────────────────────────────────────────────────
# STEP 4: CALLBACKS
#
# These run automatically during training to improve results
# ─────────────────────────────────────────────────────────────

callbacks_phase1 = [
    # Save the best model weights seen so far
    ModelCheckpoint(
        "best_model_phase1.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    # Stop training early if validation accuracy stops improving
    # patience=5 means stop after 5 epochs with no improvement
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate when validation loss plateaus
    # This helps the model converge more precisely
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,        # multiply lr by 0.5
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# ─────────────────────────────────────────────────────────────
# STEP 5: PHASE 1 TRAINING
# Train only the top classification layers
# The MobileNetV2 base stays frozen
# ─────────────────────────────────────────────────────────────

print("\n=== Phase 1: Training Top Layers (Base Frozen) ===")

history_phase1 = model.fit(
    train_generator,
    epochs=MAX_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_phase1,
    verbose=1
)

print(f"\nPhase 1 best val accuracy: {max(history_phase1.history['val_accuracy']):.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 6: PHASE 2 FINE TUNING (optional but recommended)
#
# Unfreeze the last N layers of MobileNetV2 and train at a very
# low learning rate. This allows the base model to adapt its
# features specifically to ASL hand shapes.
#
# Set FINETUNE_LAYERS = 0 at the top to skip this phase
# ─────────────────────────────────────────────────────────────

if FINETUNE_LAYERS > 0:
    print(f"\n=== Phase 2: Fine Tuning Last {FINETUNE_LAYERS} Base Layers ===")

    # Unfreeze the entire base model first
    base_model.trainable = True

    # Then re-freeze all layers except the last FINETUNE_LAYERS
    for layer in base_model.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Trainable base layers: {trainable_count} / {len(base_model.layers)}")

    # Use a much lower learning rate to avoid destroying pretrained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_phase2 = [
        ModelCheckpoint(
            "best_model_phase2.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history_phase2 = model.fit(
        train_generator,
        epochs=15,              # fewer epochs for fine tuning
        validation_data=val_generator,
        callbacks=callbacks_phase2,
        verbose=1
    )

    print(f"\nPhase 2 best val accuracy: {max(history_phase2.history['val_accuracy']):.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 7: SAVE THE FINAL MODEL
# ─────────────────────────────────────────────────────────────

print(f"\n=== Saving Model to {OUTPUT_MODEL_PATH} ===")
model.save(OUTPUT_MODEL_PATH)
print("Done. Run convert.py next to create the TFLite int8 model for ESP32.")


# ─────────────────────────────────────────────────────────────
# STEP 8: PLOT TRAINING HISTORY
# Saves a chart showing accuracy and loss over epochs
# ─────────────────────────────────────────────────────────────

print("\n=== Plotting Training History ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combine phase histories if fine tuning was done
acc = history_phase1.history["accuracy"]
val_acc = history_phase1.history["val_accuracy"]
loss = history_phase1.history["loss"]
val_loss = history_phase1.history["val_loss"]

if FINETUNE_LAYERS > 0:
    acc     += history_phase2.history["accuracy"]
    val_acc += history_phase2.history["val_accuracy"]
    loss    += history_phase2.history["loss"]
    val_loss+= history_phase2.history["val_loss"]

    # Draw a vertical line showing where fine tuning started
    phase2_start = len(history_phase1.history["accuracy"])
    axes[0].axvline(x=phase2_start, color="gray", linestyle="--", label="Fine tune start")
    axes[1].axvline(x=phase2_start, color="gray", linestyle="--", label="Fine tune start")

epochs_range = range(len(acc))

# Accuracy plot
axes[0].plot(epochs_range, acc, label="Train Accuracy")
axes[0].plot(epochs_range, val_acc, label="Val Accuracy")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# Loss plot
axes[1].plot(epochs_range, loss, label="Train Loss")
axes[1].plot(epochs_range, val_loss, label="Val Loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("Saved training_history.png")
plt.show()

print("\n=== Training Complete ===")
print(f"Model saved to   : {OUTPUT_MODEL_PATH}")
print(f"Class names saved: class_names.txt")
print(f"Training chart   : training_history.png")
print(f"\nNext step: run convert.py to generate asl_model.tflite for ESP32")

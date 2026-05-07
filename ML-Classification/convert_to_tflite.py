import numpy as np
import tensorflow as tf

MODEL_PATH   = "asl_landmark_inference_model.keras"
DATASET_PATH = "asl_landmarks_sequences.npz"
OUTPUT_PATH  = "asl_landmark_model.tflite"
NUM_CALIBRATION_SAMPLES = 200


def representative_dataset():
    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data["X"].astype("float32")
    indices = np.random.choice(len(X), size=NUM_CALIBRATION_SAMPLES, replace=False)
    for i in indices:
        yield [X[i : i + 1]]


print(f"Loading {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting to int8 TFLite...")
tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"Saved {OUTPUT_PATH}  ({size_kb:.1f} KB)")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nRunning sanity check...")
interpreter = tf.lite.Interpreter(model_path=OUTPUT_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = np.load(DATASET_PATH, allow_pickle=True)
sample = data["X"][0:1].astype("float32")  # [1, SEQUENCE_LEN, 63]

scale, zero_point = input_details[0]["quantization"]
sample_int8 = (sample / scale + zero_point).astype(np.int8)

interpreter.set_tensor(input_details[0]["index"], sample_int8)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]["index"])
print(f"Sanity check passed. Output shape: {output.shape}  argmax: {np.argmax(output)}")
print("\nConversion complete. Flash asl_landmark_model.tflite to your MCU.")

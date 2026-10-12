from pathlib import Path

import tensorflow as tf


KERAS_MODEL_PATH = Path("mnist_model.keras")
TFLITE_MODEL_PATH = Path("model.tflite")


def main() -> None:
    mnist_classifier = tf.keras.models.load_model(KERAS_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(mnist_classifier)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    TFLITE_MODEL_PATH.write_bytes(quantized_tflite_model)

    keras_model_size_kib = KERAS_MODEL_PATH.stat().st_size / 1024
    tflite_model_size_kib = TFLITE_MODEL_PATH.stat().st_size / 1024
    compression_ratio = keras_model_size_kib / tflite_model_size_kib

    print("Conversion complete")
    print(f"Keras model size: {keras_model_size_kib:,.1f} KB")
    print(f"TFLite model size: {tflite_model_size_kib:,.1f} KB")
    print(f"Keras-to-TFLite size ratio: {compression_ratio:.1f}x")


if __name__ == "__main__":
    main()

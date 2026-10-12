from pathlib import Path
import time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


TFLITE_MODEL_PATH = Path("model.tflite")
TEST_SAMPLE_PATH = Path("test_sample.npy")
BENCHMARK_RUNS = 100


def load_test_image() -> np.ndarray:
    if TEST_SAMPLE_PATH.exists():
        return np.load(TEST_SAMPLE_PATH).astype(np.float32)
    return np.random.random((1, 28, 28, 1)).astype(np.float32)


def main() -> None:
    interpreter = Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    test_image = load_test_image()

    print(f"Input shape: {input_details['shape']}")
    print(f"Output shape: {output_details['shape']}")

    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()

    latency_samples_ms = []
    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        latency_samples_ms.append((time.perf_counter() - start_time) * 1000)

    inference_scores = interpreter.get_tensor(output_details["index"])
    predicted_class = int(np.argmax(inference_scores))
    prediction_confidence = float(np.max(inference_scores))
    latency_samples_ms = np.asarray(latency_samples_ms)
    mean_latency_ms = float(latency_samples_ms.mean())

    print("\n" + "=" * 50)
    print("EDGE INFERENCE RESULTS")
    print("=" * 50)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {prediction_confidence:.4f}")
    print(f"Mean latency: {mean_latency_ms:.3f} ms")
    print(f"Median latency: {np.median(latency_samples_ms):.3f} ms")
    print(f"p95 latency: {np.percentile(latency_samples_ms, 95):.3f} ms")
    print(f"Maximum latency: {latency_samples_ms.max():.3f} ms")
    print(f"Throughput: {1000 / mean_latency_ms:.0f} inferences/second")
    print("=" * 50)
    print("Record these values in the results table")


if __name__ == "__main__":
    main()

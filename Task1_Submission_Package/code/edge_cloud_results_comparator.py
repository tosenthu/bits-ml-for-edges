import json
from pathlib import Path
import time
from typing import Tuple

import numpy as np
import requests

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


TFLITE_MODEL_PATH = Path("model.tflite")
TEST_SAMPLE_PATH = Path("test_sample.npy")
CLOUD_INFERENCE_URL = "http://127.0.0.1:8000/infer"
BENCHMARK_RUNS = 100


def load_test_image() -> np.ndarray:
    if TEST_SAMPLE_PATH.exists():
        return np.load(TEST_SAMPLE_PATH).astype(np.float32)
    return np.random.random((1, 28, 28, 1)).astype(np.float32)


def benchmark_local_inference(test_image: np.ndarray) -> Tuple[np.ndarray, int]:
    interpreter = Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()

    latency_samples_ms = []
    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        latency_samples_ms.append((time.perf_counter() - start_time) * 1000)

    inference_scores = interpreter.get_tensor(output_details["index"])
    return np.asarray(latency_samples_ms), int(np.argmax(inference_scores))


def benchmark_cloud_inference(
    test_image: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    request_payload = {"data": test_image.flatten().tolist()}
    request_body_size_bytes = len(json.dumps(request_payload).encode("utf-8"))
    requests.post(CLOUD_INFERENCE_URL, json=request_payload, timeout=10).raise_for_status()

    round_trip_samples_ms = []
    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()
        inference_response = requests.post(
            CLOUD_INFERENCE_URL,
            json=request_payload,
            timeout=10,
        )
        round_trip_samples_ms.append((time.perf_counter() - start_time) * 1000)
        inference_response.raise_for_status()

    final_response_body = inference_response.json()
    return (
        np.asarray(round_trip_samples_ms),
        int(final_response_body["prediction"]),
        request_body_size_bytes,
    )


def main() -> None:
    test_image = load_test_image()
    edge_latency_samples_ms, edge_prediction = benchmark_local_inference(test_image)
    cloud_round_trip_samples_ms, cloud_prediction, request_body_size_bytes = (
        benchmark_cloud_inference(test_image)
    )
    mean_edge_latency_ms = float(edge_latency_samples_ms.mean())
    mean_cloud_round_trip_ms = float(cloud_round_trip_samples_ms.mean())

    print("\n" + "=" * 65)
    print(f"{'METRIC':<28} {'EDGE':>16} {'CLOUD RTT':>16}")
    print("=" * 65)
    print(
        f"{'Mean latency (ms)':<28} "
        f"{mean_edge_latency_ms:>16.3f} {mean_cloud_round_trip_ms:>16.3f}"
    )
    print(
        f"{'Median latency (ms)':<28} "
        f"{np.median(edge_latency_samples_ms):>16.3f} "
        f"{np.median(cloud_round_trip_samples_ms):>16.3f}"
    )
    print(
        f"{'p95 latency (ms)':<28} "
        f"{np.percentile(edge_latency_samples_ms, 95):>16.3f} "
        f"{np.percentile(cloud_round_trip_samples_ms, 95):>16.3f}"
    )
    print(
        f"{'Throughput (inf/s)':<28} "
        f"{1000 / mean_edge_latency_ms:>16.0f} "
        f"{1000 / mean_cloud_round_trip_ms:>16.0f}"
    )
    print(f"{'Data sent per call':<28} {'0 bytes':>16} {request_body_size_bytes:>13,} B")
    print(f"{'Network required':<28} {'No':>16} {'Yes':>16}")
    print(f"{'Prediction':<28} {edge_prediction:>16} {cloud_prediction:>16}")
    print("=" * 65)
    print(
        f"\nEdge speedup: "
        f"{mean_cloud_round_trip_ms / mean_edge_latency_ms:.1f}x compared with cloud RTT"
    )
    print(
        f"Request body per inference: {request_body_size_bytes:,} bytes "
        f"({request_body_size_bytes / 1024:.1f} KB)"
    )
    print(
        f"Request bandwidth at 30 FPS: "
        f"{request_body_size_bytes * 30 / 1024:.0f} KB/s "
        f"({request_body_size_bytes * 30 / 1024 / 1024:.2f} MB/s)"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()

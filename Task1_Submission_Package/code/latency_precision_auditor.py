from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import requests

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


TFLITE_MODEL_PATH = Path("model.tflite")
TEST_SAMPLE_PATH = Path("test_sample.npy")
CLOUD_INFERENCE_URL = "http://127.0.0.1:8000/infer"
RESULTS_PATH = Path("precision-comparison-samples.json")
BENCHMARK_RUNS = 100


def load_test_image() -> np.ndarray:
    if TEST_SAMPLE_PATH.exists():
        return np.load(TEST_SAMPLE_PATH).astype(np.float32)
    return np.random.random((1, 28, 28, 1)).astype(np.float32)


def main() -> None:
    test_image = load_test_image()
    interpreter = Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()

    edge_latency_samples_ms = []
    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        edge_latency_samples_ms.append((time.perf_counter() - start_time) * 1000)
    edge_prediction = int(
        np.argmax(interpreter.get_tensor(output_details["index"]))
    )

    request_payload = {"data": test_image.flatten().tolist()}
    request_body_size_bytes = len(json.dumps(request_payload).encode("utf-8"))
    requests.post(CLOUD_INFERENCE_URL, json=request_payload, timeout=10).raise_for_status()

    cloud_round_trip_samples_ms = []
    cloud_server_latency_samples_ms = []
    cloud_prediction = None
    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()
        inference_response = requests.post(
            CLOUD_INFERENCE_URL,
            json=request_payload,
            timeout=10,
        )
        cloud_round_trip_samples_ms.append((time.perf_counter() - start_time) * 1000)
        inference_response.raise_for_status()
        response_body = inference_response.json()
        cloud_server_latency_samples_ms.append(float(response_body["server_latency_ms"]))
        cloud_prediction = int(response_body["prediction"])

    edge_latency_array_ms = np.asarray(edge_latency_samples_ms)
    cloud_round_trip_array_ms = np.asarray(cloud_round_trip_samples_ms)
    cloud_server_latency_array_ms = np.asarray(cloud_server_latency_samples_ms)
    result = {
        "runs": BENCHMARK_RUNS,
        "edge_ms": edge_latency_samples_ms,
        "server_ms": cloud_server_latency_samples_ms,
        "cloud_rtt_ms": cloud_round_trip_samples_ms,
        "summary": {
            "edge_mean_ms": float(edge_latency_array_ms.mean()),
            "edge_median_ms": float(np.median(edge_latency_array_ms)),
            "edge_p95_ms": float(np.percentile(edge_latency_array_ms, 95)),
            "edge_max_ms": float(edge_latency_array_ms.max()),
            "edge_throughput_per_second": float(1000 / edge_latency_array_ms.mean()),
            "server_mean_ms": float(cloud_server_latency_array_ms.mean()),
            "server_p95_ms": float(np.percentile(cloud_server_latency_array_ms, 95)),
            "cloud_rtt_mean_ms": float(cloud_round_trip_array_ms.mean()),
            "cloud_rtt_median_ms": float(np.median(cloud_round_trip_array_ms)),
            "cloud_rtt_p95_ms": float(np.percentile(cloud_round_trip_array_ms, 95)),
            "cloud_rtt_max_ms": float(cloud_round_trip_array_ms.max()),
            "cloud_throughput_per_second": float(
                1000 / cloud_round_trip_array_ms.mean()
            ),
            "edge_advantage_ms": float(
                cloud_round_trip_array_ms.mean() - edge_latency_array_ms.mean()
            ),
            "edge_speedup_factor": float(
                cloud_round_trip_array_ms.mean() / edge_latency_array_ms.mean()
            ),
            "mean_non_inference_overhead_ms": float(
                cloud_round_trip_array_ms.mean() - cloud_server_latency_array_ms.mean()
            ),
            "payload_bytes": request_body_size_bytes,
            "bandwidth_30_fps_kib_per_second": request_body_size_bytes * 30 / 1024,
            "edge_prediction": edge_prediction,
            "cloud_prediction": cloud_prediction,
            "predictions_match": edge_prediction == cloud_prediction,
        },
    }

    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Raw samples written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

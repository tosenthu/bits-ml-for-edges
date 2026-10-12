import json
from pathlib import Path
import sys
import time

import numpy as np
import requests


CLOUD_INFERENCE_URL = "http://127.0.0.1:8000/infer"
CLOUD_HEALTH_URL = "http://127.0.0.1:8000/"
TEST_SAMPLE_PATH = Path("test_sample.npy")
BENCHMARK_RUNS = 100


def load_test_image() -> np.ndarray:
    if TEST_SAMPLE_PATH.exists():
        return np.load(TEST_SAMPLE_PATH).astype(np.float32)
    return np.random.random((1, 28, 28, 1)).astype(np.float32)


def main() -> None:
    test_image = load_test_image()
    request_payload = {"data": test_image.flatten().tolist()}
    request_body_size_bytes = len(json.dumps(request_payload).encode("utf-8"))
    print(
        f"Request body size: {request_body_size_bytes:,} bytes "
        f"({request_body_size_bytes / 1024:.1f} KB)"
    )

    try:
        health_response = requests.get(CLOUD_HEALTH_URL, timeout=5)
        health_response.raise_for_status()
    except requests.RequestException as error:
        print(f"Cloud server is unavailable: {error}")
        print("Start cloud_inference_server.py before running this client")
        raise SystemExit(1) from error

    print("Server reachable")
    requests.post(CLOUD_INFERENCE_URL, json=request_payload, timeout=10).raise_for_status()

    round_trip_samples_ms = []
    server_latency_samples_ms = []
    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()
        inference_response = requests.post(
            CLOUD_INFERENCE_URL,
            json=request_payload,
            timeout=10,
        )
        round_trip_samples_ms.append((time.perf_counter() - start_time) * 1000)
        inference_response.raise_for_status()
        response_body = inference_response.json()
        server_latency_samples_ms.append(float(response_body["server_latency_ms"]))

    round_trip_samples_ms = np.asarray(round_trip_samples_ms)
    server_latency_samples_ms = np.asarray(server_latency_samples_ms)
    final_response_body = inference_response.json()
    network_overhead_samples_ms = round_trip_samples_ms - server_latency_samples_ms

    print("\n" + "=" * 50)
    print("CLOUD INFERENCE RESULTS")
    print("=" * 50)
    print(f"Prediction: {final_response_body['prediction']}")
    print(f"Mean server inference: {server_latency_samples_ms.mean():.3f} ms")
    print(f"Server p95 latency: {np.percentile(server_latency_samples_ms, 95):.3f} ms")
    print(f"Mean round-trip time: {round_trip_samples_ms.mean():.3f} ms")
    print(f"Round-trip p95 latency: {np.percentile(round_trip_samples_ms, 95):.3f} ms")
    print(f"Mean overhead outside inference: {network_overhead_samples_ms.mean():.3f} ms")
    print(f"Request body size: {request_body_size_bytes:,} bytes")
    print(f"Request bandwidth at 30 FPS: {request_body_size_bytes * 30 / 1024:.0f} KB/s")
    print("=" * 50)
    print("Record these values in the results table")


if __name__ == "__main__":
    main()

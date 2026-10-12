# Task 1 Execution Logs

This directory contains the captured terminal transcripts and full-precision benchmark data used as evidence for Task 1. The numbered transcripts follow the experiment sequence, with two additional checks demonstrating the difference between cloud and local network dependency.

## Transcript index

| File | Contents |
|---|---|
| `01-build-model.txt` | MNIST dataset counts, CNN model summary, five training epochs, test-set accuracy, and confirmation that the trained model and fixed test sample were saved. |
| `02-convert-to-tflite.txt` | TensorFlow Lite conversion output, including the reported source-model size, optimized TFLite size, and compression ratio. |
| `03-export-saved-model.txt` | Supplementary SavedModel export and the logical size of the resulting directory. |
| `04-local-inference.txt` | Local TensorFlow Lite input/output shapes, prediction, confidence, and 100-run edge latency and throughput statistics. |
| `05-cloud-server.txt` | FastAPI inference server initialization, model readiness on port 8000, and the later manual server stop. |
| `06-cloud-client.txt` | Request payload size, server reachability, cloud prediction, server compute timing, total round-trip timing, network overhead, and the 30-FPS bandwidth estimate. |
| `07-compare-results.txt` | Unified edge-versus-cloud table, matching predictions, edge speedup, payload size, and bandwidth calculations. |
| `08-precision-audit.txt` | Supplementary full-precision summary for edge, server, and cloud RTT measurements, including derived overhead, speedup, bandwidth, and prediction agreement. |
| `09-cloud-client-server-down.txt` | Negative-control check showing that the cloud client cannot infer after the cloud server is stopped. |
| `10-local-after-server-down.txt` | Follow-up check showing that local TensorFlow Lite inference still succeeds after the cloud server is stopped. |

## Full-precision data

`precision-comparison-samples.json` contains all 100 unrounded edge-latency samples, all 100 server-compute samples, all 100 cloud round-trip samples, and a calculated summary. It supports independent verification of means, medians, percentiles, throughput, edge advantage, speedup, non-inference overhead, payload bandwidth, and edge/cloud prediction agreement without relying on values rounded for terminal display.

## Raw-log display note

These files are raw terminal captures. They may contain ANSI colour sequences, carriage returns, backspace characters, or other terminal control characters—particularly in TensorFlow/Keras progress output. Some text viewers may therefore show symbols such as `^[`, `^D`, or overwritten progress lines. The characters were retained because the transcripts serve as raw execution evidence; the assignment report provides the cleaner human-readable presentation.

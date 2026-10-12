# Lab Worksheet 1: MNIST Edge and Cloud Inference

**Course:** Machine Learning on the Edge  
**Student:** Senthil Kumar M R  
**Student ID:** 2025cs05049  
**Experiment run date:** 22 September 2026

## Aim

This experiment compares a TensorFlow Lite model running locally with the same model called through a FastAPI service. The comparison looks at accuracy, model size, latency, bandwidth, and what happens when the server is unavailable.

## Setup and method

The supplied Python programs were run in this order: train the MNIST model, convert it to TensorFlow Lite, benchmark local inference, start the API server and benchmark its client, then run the combined comparison. The cloud-style server and client ran on the same computer using `localhost`; this is a REST benchmark, not a measurement of a remote cloud connection.

The run used a MacBook Pro (Mac15,7) with an Apple M3 Pro processor and 18 GB of memory. The environment reported Python 3.13.5, TensorFlow 2.21.0, Keras 3.14.1, and FastAPI 0.136.1. The local timing uses the laptop CPU with the XNNPACK delegate, so it represents a laptop simulation of edge inference rather than a microcontroller benchmark.

## Model training

The model was trained for five epochs on 60,000 MNIST training images and evaluated on 10,000 test images. It used Adam, sparse categorical cross-entropy, batch size 64, and a 10% validation split.

| Layer | Output shape | Parameters |
|---|---:|---:|
| Conv2D, 8 filters, 3×3, ReLU | 28×28×8 | 80 |
| Max pooling, 2×2 | 14×14×8 | 0 |
| Flatten | 1,568 | 0 |
| Dense, 32 units, ReLU | 32 | 50,208 |
| Dense, 10 units, softmax | 10 | 330 |
| **Total** | — | **50,618** |

| Training result | Value |
|---|---:|
| Final training accuracy | 97.90% |
| Final training loss | 0.0688 |
| Final validation accuracy | 98.27% |
| Final validation loss | 0.0631 |
| Test accuracy | **97.85%** |

The test result is close to the final training accuracy, so this run does not show a large train/test gap. The supplied training program does not set a random seed; a later training run may produce slightly different values. The benchmark image was test sample 0, whose true label was 7.

## TensorFlow Lite conversion

The converter used `Optimize.DEFAULT`, which applies dynamic-range weight optimization. The inspected model interface still takes and returns `float32`; it is not a fully integer-only input/output model. The larger dense weight matrix was stored in INT8 form.

| Artifact | Size |
|---|---:|
| Saved Keras training model (`mnist_model.keras`) | 637,507 bytes (622.6 KiB) |
| Exported inference SavedModel directory | 452,678 bytes (442.1 KiB) |
| TensorFlow Lite model (`model.tflite`) | 55,960 bytes (54.6 KiB) |

The supplied script saves a `.keras` file even though its console text calls it a SavedModel. I also exported the trained model to a SavedModel directory, without retraining, to measure the directory requested in the worksheet. Using that directory, the SavedModel/TFLite ratio is `452,678 ÷ 55,960 = 8.09×`, a size reduction of 87.64%. The supplied conversion script separately reports the Keras/TFLite ratio as 11.4×.

## Local inference

The TFLite interpreter performed one warm-up call, then timed 100 inferences on the saved test image. The warm-up was excluded.

| Result | Value |
|---|---:|
| True label | 7 |
| Predicted label | 7 |
| Confidence | 0.9996 |
| Mean latency | 0.006 ms |
| Median latency | 0.006 ms |
| 95th-percentile latency | 0.007 ms |
| Maximum latency | 0.009 ms |
| Throughput reported by script | 161,335 inferences/s |

The model predicted the saved image correctly. The timing values are close together in this 100-run sample. This local path does not send the image over HTTP. The measured time covers interpreter invocation, not image capture or all application work.

## Cloud-style inference

The FastAPI server loaded the same TFLite model. The client sent the same saved image in 100 sequential HTTP requests. The supplied code uses the `/infer` endpoint, although the course announcement refers to `/predict`.

| Result | Value |
|---|---:|
| Prediction | 7 |
| Mean server inference time | 0.012 ms |
| Server 95th-percentile time | 0.016 ms |
| Mean client round-trip time | 1.646 ms |
| Client round-trip 95th percentile | 2.006 ms |
| Mean round-trip time outside measured inference | 1.634 ms |
| Serialized JSON request body | 5,687 bytes |
| Request-body rate at 30 FPS | 166.6 KiB/s (about 167 KB/s as printed) |

Most of the measured localhost round trip was outside the model's inference call. This includes JSON handling, HTTP/API processing, local transfer, scheduling, and response parsing. The 5,687-byte figure is the request body only; it excludes the response, HTTP headers, and network transport overhead. Because the client and server shared one computer, these numbers do not represent internet latency.

## Combined comparison

| Measurement | Value | Network dependency | Note |
|---|---:|---|---|
| Local TFLite mean latency | 0.007 ms | No | Local interpreter compute |
| Server compute mean | 0.012 ms | Server | Reported in the JSON response |
| Cloud-style mean round trip | 1.502 ms | Yes | Localhost HTTP request and response |
| Edge advantage from displayed means | 1.495 ms | — | RTT minus local mean |
| Speedup reported by supplied script | 226.8× | — | Uses the run's unrounded timing arrays |
| JSON request body per call | 5,687 bytes | — | Request body only |
| Edge and server predictions match | Yes | — | Both predicted digit 7 |

The combined run found lower latency on the local path. A supplementary run retained the unrounded timing samples and measured 0.006678269 ms mean edge time and 1.430055039 ms mean round-trip time, a 214.135581× ratio. The timing changed slightly between runs, which is expected for short measurements on a shared operating system.

## Reflection questions

### R1 — Latency and real-time constraints

At 600 parts per minute, each part arrives every 100 ms. Both the measured local mean (0.007 ms) and the localhost round trip (1.502 ms) are below that interval. The comparison does not show how a remote cloud service would perform because no WAN connection was measured. For a vehicle with a 10 ms pedestrian-detection deadline, a remote round trip can be unpredictable, so the time-critical decision should run on the vehicle; cloud processing can support non-critical analysis or updates.

### R2 — Privacy and data sovereignty

In the cloud test, the image values leave the client as a JSON request. Similar data exposure can arise in patient monitoring (HIPAA), home cameras or voice assistants (GDPR Article 25), and workplace or retail biometric monitoring (PDPA). If inference runs on the device, raw images, audio, or sensor values can stay there and only a minimal result can be sent onward. This reduces the raw-data transfer and storage path, though it does not remove every privacy or security obligation.

### R3 — Bandwidth economics

Using the question's assumptions of 500 cameras, 25 frames per second, and 50 KB per frame:

```text
Cloud video: 500 × 25 × 50 KB = 625,000 KB/s = 625 MB/s (about 5 Gbit/s)
Edge labels: 500 × 25 × 1 byte = 12,500 bytes/s = 12.5 KB/s
Difference: approximately 624.99 MB/s less uplink traffic
```

This assumes every keyframe is transmitted and each edge result is one byte. The reduction would lower network and cloud-ingress requirements. A system might still send selected clips for review, but it would not need to stream every frame just to get a label.

### R4 — Connectivity and resilience

When the server was stopped, the client reported that the server must be started first. The local inference script still loaded the TFLite model and predicted 7. This shows the cloud path depends on the server and network path, while the local model can keep working without that service, as long as the device itself is available. Examples where offline decisions matter include robot or vehicle collision stopping, remote industrial or agricultural sites, and emergency-response equipment.

### R5 — Edge–cloud continuum

| Scenario | Suggested tier | Main constraint | Reason |
|---|---|---|---|
| Always-on keyword spotting | Edge | Privacy | Audio can be processed locally and responses are immediate. |
| Monthly patient-outcome analysis | Private cloud | Privacy | Batch work is not latency-sensitive; access and data handling need controls. |
| Robot-arm collision avoidance | Edge | Latency | The safety decision belongs in the local control loop. |
| Payment-terminal fraud checks | Fog, with local fallback | Connectivity | A nearby service can use broader context while the terminal can handle temporary disconnection. |

## Limitations

- The cloud-style test used `localhost`, not a remote cloud provider.
- The local benchmark used a high-performance laptop rather than a low-power edge board.
- The timing comparison used one saved image and 100 sequential runs.
- The JSON size excludes response and transport overhead.
- The training script does not fix a random seed.
- The TFLite model was not evaluated over the full 10,000-image test set in this run.

## Supporting files

The accompanying package includes the six supplied Python programs, the trained Keras and TFLite models, the saved benchmark sample, the supplemental SavedModel export used for the size row, a JSON result summary, and the captured terminal logs. The logs are raw terminal captures and may show progress-control characters. They support the values reported above.

# Task 1 Source Code

These scripts implement the MNIST training, TensorFlow Lite conversion, local inference, and cloud-style comparison. The filenames describe each script's role, and the variable names follow standard Python `snake_case` conventions.

## Run order

Run the commands from this directory so the generated model files and benchmark sample stay together.

```bash
python tensorflow_model_builder.py
python tflite_model_converter.py
python saved_model_exporter.py
python local_inference_benchmarker.py
```

Start the inference service in one terminal:

```bash
python cloud_inference_server.py
```

In another terminal, run the client and combined comparison:

```bash
python cloud_inference_client.py
python edge_cloud_results_comparator.py
python latency_precision_auditor.py
```

The service listens on `127.0.0.1` because this experiment simulates the cloud endpoint locally. Its request model accepts one 28 × 28 image with normalized pixel values from 0 to 1. The benchmark uses 100 measured inferences after a warm-up call.

`saved_model_exporter.py` is supplementary: it exports the trained Keras model to a SavedModel directory for the requested size measurement. It stops if the destination already exists, rather than deleting existing files. `latency_precision_auditor.py` is also supplementary; it writes the unrounded timing samples and derived metrics to `precision-comparison-samples.json`.

The captured logs in `supporting/logs/` are retained as evidence from the earlier run and therefore contain the original worksheet script names. The renamed scripts are the maintainable source versions; the logs have not been edited or relabelled.

## Dependencies

```text
tensorflow>=2.13
fastapi
uvicorn
requests
numpy
psutil
```

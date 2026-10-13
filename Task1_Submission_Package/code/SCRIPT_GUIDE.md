# Script guide

Run these commands from this folder. They create the model files and test image here.

The MNIST dataset is included in `../supporting/data/mnist.npz`. The training script reads this local file, so it does not need to download MNIST or rely on local SSL certificates.

```bash
python tensorflow_model_builder.py
python tflite_model_converter.py
python saved_model_exporter.py
python local_inference_benchmarker.py
```

Start the local server in one terminal:

```bash
python cloud_inference_server.py
```

In another terminal, send test requests and compare the results:

```bash
python cloud_inference_client.py
python edge_cloud_results_comparator.py
python latency_precision_auditor.py
```

The server listens on this computer only (`127.0.0.1`). It accepts one 28 × 28 image with pixel values between 0 and 1. The benchmark makes one warm-up prediction, then measures 100 predictions.

`saved_model_exporter.py` creates the SavedModel folder for the size comparison. It stops if that folder already exists. `latency_precision_auditor.py` saves the exact timing samples and calculated results as `precision-comparison-samples.json`.

The logs were captured before the source files were renamed, so they show the worksheet's original file names. The logs are left unchanged.

Install the packages listed in `requirements.txt` before running the scripts.

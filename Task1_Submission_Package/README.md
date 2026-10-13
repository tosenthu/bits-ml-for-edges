# Running the experiment

This package contains the report, code, saved model files, and results for Task 1. To run the experiment again, unzip the package and open Terminal in its `code` folder, where `requirements.txt` is located.

## Set up Python

The package uses `requirements.txt`, not `pyproject.toml`. So, do not use `uv add`. Use either setup below.

With Python and pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Or with `uv`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The MNIST dataset is included in `supporting/data/mnist.npz`, so training does not need to download data or depend on the computer's SSL certificate setup. Internet access is needed only to install packages that are not already available.

## Train and run local inference

Run these commands from the `code` folder, using the Python environment you set up above:

```bash
python tensorflow_model_builder.py
python tflite_model_converter.py
python saved_model_exporter.py
python local_inference_benchmarker.py
```

If using `uv` without activating the environment, prefix each command with `.venv/bin/`; for example, `.venv/bin/python tensorflow_model_builder.py`.

## Run the cloud-style test

Start the server in one Terminal window and leave it running:

```bash
python cloud_inference_server.py
```

In a second Terminal window, go to the same `code` folder, activate `.venv` if using pip, and run:

```bash
python cloud_inference_client.py
python edge_cloud_results_comparator.py
python latency_precision_auditor.py
```

The server runs on this computer at `127.0.0.1`; it is a local web-service test, not a remote cloud service. A new training or timing run may produce slightly different results from those recorded in the report.

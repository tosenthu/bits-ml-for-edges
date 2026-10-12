# Lab Worksheet 1: MNIST Edge and Cloud Inference

**Course:** Machine Learning on the Edge · **Student:** Senthil Kumar M R · **Student ID:** 2025cs05049 · **Experiment date:** 22 September 2026

## Aim

In this experiment, I trained a small model to recognise handwritten digits. I then compared its prediction time when it ran on my computer with the time taken when the same model was called through a web service. I also recorded the model sizes and the amount of data sent to the service.

## Simple meaning of a few terms

- **Model:** A program that learns patterns from examples and uses them to make a prediction.
- **Training:** Showing the model many labelled examples so it can learn those patterns.
- **Inference:** Asking a trained model to make a prediction for new input.
- **Latency:** The time taken to return one prediction.
- **Throughput:** The number of predictions that can be made in one second.
- **Speedup:** How many times one method is faster than another in the measured test.
- **Edge inference:** Running the model on the device that collects or uses the data.
- **Cloud inference:** Sending data to a server and getting the prediction back.

## Setup and steps

I used the MNIST dataset, which contains pictures of handwritten digits from 0 to 9. The program trained a small convolutional neural network (CNN), a type of model that can find simple shapes and lines in an image. I converted it to TensorFlow Lite and tested it on one saved image. The same image was then sent to a FastAPI web service for the cloud-style test.

The source files in the package have clear names. They follow the same steps as the worksheet programs:

1. `tensorflow_model_builder.py` trains the model and saves a test image.
2. `tflite_model_converter.py` converts the trained model to TensorFlow Lite.
3. `local_inference_benchmarker.py` measures predictions on this computer.
4. `cloud_inference_server.py` starts the web service; `cloud_inference_client.py` sends the image to it.
5. `edge_cloud_results_comparator.py` runs the local and server tests and compares them.

The saved terminal logs and measurements are from the original worksheet run, so they still show the worksheet's original file names. I cleaned up the source names and variable names after that run. I did not repeat the full timing test after the clean-up. I did check that the updated server accepts a valid image and rejects pixel values outside the expected range.

The server and client ran on the same MacBook using `localhost` (this computer). So this is a local web-service test, not a test over the internet or to a real cloud server.

| Computer or software | Used in this experiment |
|---|---|
| Computer | MacBook Pro, Mac15,7 |
| Processor and memory | Apple M3 Pro, 18 GB |
| Python | 3.13.5 |
| TensorFlow / Keras | 2.21.0 / 3.14.1 |
| FastAPI | 0.136.1 |
| Local inference support | XNNPACK, a helper for running predictions on the CPU |

## Model training

The model learned from 60,000 images and was checked using 10,000 separate test images. Training ran for five epochs (five rounds over the training examples). I used a batch size of 64 and kept 10% of the training images for checking progress during training. Accuracy is the percentage of images classified correctly. The loss value shows how far the model's answers were from the correct answers; a lower loss is better. Parameters are the values the model adjusts while learning.

| Model layer | What it does | Number of parameters |
|---|---|---:|
| Convolution, 8 filters, 3×3, ReLU | Finds small lines and shapes in the digit image | 80 |
| 2×2 max pooling | Shrinks the image features while keeping important details | 0 |
| Flatten | Changes the image features into one long list | 0 |
| Dense, 32 units, ReLU | Combines features to help identify the digit | 50,208 |
| Dense, 10 units, softmax | Gives a score for each digit from 0 to 9 | 330 |
| **Total** |  | **50,618** |

| Training check | Result |
|---|---:|
| Accuracy on training images at the end | 97.90% |
| Accuracy on validation images | 98.27% |
| Accuracy on the separate test set | **97.85%** |
| Training loss | 0.0688 |
| Validation loss | 0.0631 |

The model correctly classified most test images. The test accuracy is close to the training accuracy, so this run does not show a large difference between the images used for learning and the separate test images. The training program does not fix a random seed, so training again can give slightly different results. The saved example image was the first test image. Its correct label was 7.

## Converting the model to TensorFlow Lite

TensorFlow Lite is a model format used for running models on phones and other smaller devices. I used `Optimize.DEFAULT`. This option reduces the size of some model weights. The weights are stored numbers that the model learned during training.

The converted model still accepts and returns `float32` values, so it is not a fully integer model. A `float32` value uses 32 bits to store a number. Some of the larger weights were stored using INT8, which uses 8 bits and takes less space.

| File or folder | Size |
|---|---:|
| Trained Keras model, `mnist_model.keras` | 637,507 bytes (622.6 KiB) |
| Exported inference SavedModel folder | 452,678 bytes (442.1 KiB) |
| TensorFlow Lite model, `model.tflite` | 55,960 bytes (54.6 KiB) |

The worksheet asks for the size of a SavedModel folder. The training program saves a `.keras` file, so I also exported that same trained model to a SavedModel folder. I did not train the model again for this export.

The SavedModel folder is about **8.09 times** the size of the TensorFlow Lite file. In other words, the TensorFlow Lite file is about **87.64% smaller** than that folder. The original conversion script also compares the Keras file with the TensorFlow Lite file; that ratio is 11.4 times. These are two different comparisons, so their ratios are not the same.

## Local inference on this computer

The local program loaded the TensorFlow Lite model, made one warm-up prediction, and then measured 100 predictions using the saved image. The warm-up was not included in the results.

| Result | Measured value |
|---|---:|
| Correct digit | 7 |
| Predicted digit | 7 |
| Confidence score | 0.9996 |
| Average time for one prediction | 0.006 ms |
| Middle value (median) | 0.006 ms |
| 95th-percentile time | 0.007 ms |
| Longest time in this test | 0.009 ms |
| Throughput shown by the program | 161,335 predictions per second |

The model predicted the saved image correctly. Its confidence score is close to 1, which means it was very sure about this prediction. The mean is the average time over all 100 predictions. The 95th percentile means that 95 out of the 100 measured times were at or below that value. These times cover model prediction only; they do not include taking a photo, preparing the image, or showing the result on a screen.

This test ran on a powerful laptop, not on a small device such as a microcontroller. So these times show how the model runs on this laptop; they should not be treated as expected times for every edge device.

## Cloud-style inference using the local server

The server loaded the same TensorFlow Lite model. The client sent the same image as 784 numbers in a JSON request (a text format for sending data). It first sent one warm-up request and then measured 100 requests. The submitted code uses the `/infer` address. The course announcement mentions `/predict`, but the supplied worksheet programs use `/infer`.

| Result | Measured value |
|---|---:|
| Predicted digit | 7 |
| Average time spent by the model on the server | 0.012 ms |
| Server 95th-percentile time | 0.016 ms |
| Average time for the full request and response | 1.646 ms |
| 95th-percentile full request time | 2.006 ms |
| Average time outside the model prediction | 1.634 ms |
| JSON request body size | 5,687 bytes |
| Request data rate at 30 images per second | 166.6 KiB/s (about 167 KB/s) |

The server's model calculation took only a small part of the full request time. The rest included preparing the JSON, handling the web request, sending it over the computer's local connection, and reading the response. The 5,687-byte size is only the request body. It does not include the response or internet transport information. As the server and client were on the same computer, these results do not show internet or remote-cloud delay.

## Combined edge and cloud comparison

The comparison program measured local inference and the full request to the server in another run. The results below are copied from that run.

| Measurement | Value | Needs a network? | What it means |
|---|---:|---|---|
| Average local prediction time | 0.007 ms | No | Model runs on this computer |
| Average server model time | 0.012 ms | Server connection | Time spent on the server's model calculation |
| Average full server request time | 1.502 ms | Yes | Sending the request and receiving its answer over localhost |
| Difference between full server time and local time | 1.495 ms | — | Extra time in this comparison run |
| Speedup printed by the comparison program | 226.8× | — | Uses more exact times than the rounded values shown above |
| Throughput in the combined run | 151,001 predictions/s | 666 requests/s | Calculated from average latency |
| JSON request body per image | 5,687 bytes | — | Text data sent for one request, before network overhead |
| Do local and server predictions match? | Yes | — | Both predicted digit 7 |

The comparison run and the separate cloud run gave different average times because they were measured at different times on a shared computer. Short measurements can change slightly due to other work happening on the computer. A supplementary timing check retained the exact timing samples and gave an edge time of 0.006678 ms and a round-trip time of 1.430055 ms, or about 214.14 times faster for edge inference in that run.

The main result is that local prediction was faster in these tests and did not need to send the image to a server. The cloud-style test still ran on this same laptop, so the results cannot tell us how fast a real remote cloud service would be.

## Reflection questions

### R1 — Latency and real-time use

At 600 parts per minute, one part arrives every 100 milliseconds. The measured local time (0.007 ms) and the full request time to the local server (1.502 ms) are both below 100 ms. However, the server test used `localhost`, so it does not include real internet delay. For a vehicle that must react within 10 milliseconds to avoid a person, it is safer to make the urgent decision inside the vehicle. The cloud can still help with work that is not urgent, such as later analysis or model updates.

### R2 — Privacy and data protection

In the server test, the image values were sent from the client to the server as JSON. In an actual application, images, voice recordings, or sensor readings can contain private information. Running the model on the device can keep the original data on that device and send only a short result, if needed. This can reduce the amount of private data being shared, but the application still needs proper security and data protection.

### R3 — Data use and bandwidth

Using the question's example of 500 cameras, 25 images per second from each camera, and 50 KB per image:

```text
Sending every image: 500 × 25 × 50 KB = 625,000 KB/s, or about 625 MB/s
Sending one-byte labels: 500 × 25 × 1 byte = 12,500 bytes/s, or about 12.5 KB/s
```

In this example, sending only one-byte labels would reduce the amount of data sent by about 624.99 MB every second. This could reduce the network capacity and cloud storage needed. A system may still send selected images for review, but it would not have to send every image just to get a prediction. This calculation assumes each result is one byte and does not include network overhead.

### R4 — What happens when the connection is lost?

When the server was stopped, the recorded client run could not connect and asked for the server to be started. The local program still loaded the model and predicted digit 7. This shows that local prediction can continue without the server, as long as the device itself is working. This is useful for robots, vehicles, farms, factories, and emergency equipment where a network may be slow or unavailable.

### R5 — Choosing where to run the model

| Example | Suggested place to run it | Main reason |
|---|---|---|
| Always-on keyword detection | On the device (edge) | Voice can stay on the device and the response is quick. |
| Monthly analysis of patient results | Private cloud | It can process many records together and does not need an instant answer. |
| Robot arm collision warning | On the robot (edge) | The robot must react immediately. |
| Payment fraud check | Nearby server with local backup | A nearby server can check more information; local backup helps if the connection drops. |

## Limitations

- The server and client ran on the same laptop, not on a real cloud service.
- The local test used a high-performance laptop, not a low-power edge device.
- The timing test used one image and 100 measured runs.
- The request size does not include the response or network overhead.
- The training program does not fix a random seed, so a new training run may give slightly different results.
- This run did not measure accuracy across all 10,000 test images using the converted TensorFlow Lite model.
- The full benchmark was not repeated after the source files were renamed and cleaned up; the reported timing values are from the saved run logs.

## Supporting files

The package includes the trained Keras model, the TensorFlow Lite model, the exported SavedModel folder, the saved test image, a results file, and the original run logs. The logs may contain control characters from the terminal's progress display. They have been kept as captured and support the values in this report.

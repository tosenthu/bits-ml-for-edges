from typing import Dict, List, Union
import time

from fastapi import FastAPI
from pydantic import BaseModel, validator
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


IMAGE_PIXEL_COUNT = 28 * 28
TFLITE_MODEL_PATH = "model.tflite"

app = FastAPI(title="Lab 1 Cloud Inference")
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


class InferenceRequest(BaseModel):
    data: List[float]

    @validator("data")
    def validate_normalized_image(cls, pixel_values: List[float]) -> List[float]:
        if len(pixel_values) != IMAGE_PIXEL_COUNT:
            raise ValueError(f"Expected exactly {IMAGE_PIXEL_COUNT} pixel values")
        if any(
            not np.isfinite(pixel_value) or not 0.0 <= pixel_value <= 1.0
            for pixel_value in pixel_values
        ):
            raise ValueError("Pixel values must be finite numbers between 0 and 1")
        return pixel_values


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/infer")
def run_inference(request_body: InferenceRequest) -> Dict[str, Union[int, float]]:
    image_tensor = np.asarray(request_body.data, dtype=np.float32).reshape(1, 28, 28, 1)
    start_time = time.perf_counter()
    interpreter.set_tensor(input_details["index"], image_tensor)
    interpreter.invoke()
    server_latency_ms = (time.perf_counter() - start_time) * 1000

    inference_scores = interpreter.get_tensor(output_details["index"])
    predicted_class = int(np.argmax(inference_scores))
    prediction_confidence = float(np.max(inference_scores))

    return {
        "prediction": predicted_class,
        "confidence": round(prediction_confidence, 4),
        "server_latency_ms": round(server_latency_ms, 3),
        "payload_bytes": len(request_body.data) * 4,
    }


if __name__ == "__main__":
    import uvicorn

    print("Model is ready on 127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

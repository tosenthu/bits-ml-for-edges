from pathlib import Path

import tensorflow as tf


KERAS_MODEL_PATH = Path("mnist_model.keras")
SAVED_MODEL_PATH = Path("mnist_saved_model")


def main() -> None:
    if SAVED_MODEL_PATH.exists():
        raise FileExistsError(
            f"{SAVED_MODEL_PATH} already exists; move it or choose a new output path"
        )

    mnist_classifier = tf.keras.models.load_model(KERAS_MODEL_PATH)
    mnist_classifier.export(SAVED_MODEL_PATH)
    saved_model_size_bytes = sum(
        artifact_path.stat().st_size
        for artifact_path in SAVED_MODEL_PATH.rglob("*")
        if artifact_path.is_file()
    )
    print(f"SavedModel directory: {SAVED_MODEL_PATH}")
    print(
        f"SavedModel logical size: {saved_model_size_bytes:,} bytes "
        f"({saved_model_size_bytes / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()

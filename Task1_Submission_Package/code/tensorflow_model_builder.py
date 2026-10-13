from pathlib import Path

import numpy as np
import tensorflow as tf


TRAINING_EPOCHS = 5
TRAINING_BATCH_SIZE = 64
MNIST_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "supporting" / "data" / "mnist.npz"
)
MODEL_PATH = Path("mnist_model.keras")
TEST_SAMPLE_PATH = Path("test_sample.npy")


def main() -> None:
    if not MNIST_DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"MNIST data file not found: {MNIST_DATASET_PATH}. "
            "Extract the full submission package, including supporting/data/mnist.npz."
        )

    with np.load(MNIST_DATASET_PATH) as mnist_dataset:
        training_images = mnist_dataset["x_train"]
        training_labels = mnist_dataset["y_train"]
        test_images = mnist_dataset["x_test"]
        test_labels = mnist_dataset["y_test"]

    print(f"Loaded MNIST data from {MNIST_DATASET_PATH}")

    training_images = training_images.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

    print(f"Training samples: {len(training_images):,}")
    print(f"Test samples: {len(test_images):,}")

    mnist_classifier = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                activation="relu",
                padding="same",
                input_shape=(28, 28, 1),
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="softmax"),
        ],
        name="mnist_cnn",
    )
    mnist_classifier.summary()

    mnist_classifier.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    mnist_classifier.fit(
        training_images,
        training_labels,
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_accuracy = mnist_classifier.evaluate(
        test_images,
        test_labels,
        verbose=0,
    )
    print(f"\nTest accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Test loss: {test_loss:.4f}")

    mnist_classifier.save(MODEL_PATH)
    np.save(TEST_SAMPLE_PATH, test_images[:1])
    print(f"Model saved to {MODEL_PATH}")
    print(f"Benchmark sample saved to {TEST_SAMPLE_PATH} (true label: {test_labels[0]})")


if __name__ == "__main__":
    main()

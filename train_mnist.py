import tensorflow as tf
from tensorflow.keras.datasets import mnist
import argparse
import pandas as pd


def train_model(output_file, epochs, batch_size):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

    # Define a simple CNN model -> Convolutional Neural Network (CNN)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="sigmoid"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    # Save model to the output file path (e.g., "results/model.h5")
    model.save(output_file)

    # Save model accuracy and loss to a separate text file (e.g., "results/model_output.txt")
    # loss, acc = model.evaluate(x_test, y_test)
    # output_txt = output_file.replace(".h5", "_output.txt")
    # with open(output_txt, "w") as f:
    #     f.write(f"Loss: {loss}\nAccuracy: {acc}")

    # Save loss, accuracy, and validation metrics to a plain text file
    output_txt_file = output_file.replace(".h5", "_output.txt")
    with open(output_txt_file, "w") as f:
        for epoch in range(epochs):
            f.write(f"Epoch {epoch+1}\n")
            f.write(f"Loss: {history.history['loss'][epoch]}\n")
            if "val_loss" in history.history:
                f.write(f"Validation Loss: {history.history['val_loss'][epoch]}\n")
            f.write(f"Accuracy: {history.history['accuracy'][epoch]}\n")
            if "val_accuracy" in history.history:
                f.write(
                    f"Validation Accuracy: {history.history['val_accuracy'][epoch]}\n"
                )
            f.write("\n")


# Add argument parsing for epochs and batch_size
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=3,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=128,
        help="Batch size for training",
    )
    args = parser.parse_args()

    # Call train_model with the parsed arguments
    train_model(args.output, args.epochs, args.batch_size)

"""Seven-class facial-expression CNN; saves genuine validation results."""
import argparse
import json
from pathlib import Path

def train(data_dir, output_dir, epochs=15, batch_size=32):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf.keras.utils.set_random_seed(42)
    root = Path(data_dir)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(classes) != 7:
        raise ValueError(f"Expected seven class folders, got {classes}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    opts = dict(directory=str(root), validation_split=.2, seed=42,
                image_size=(96, 96), batch_size=batch_size, label_mode="int")
    training = keras.utils.image_dataset_from_directory(
        subset="training", shuffle=True, **opts)
    validation = keras.utils.image_dataset_from_directory(
        subset="validation", shuffle=False, **opts)
    names = training.class_names
    model = keras.Sequential([
        layers.Input(shape=(96, 96, 3)),
        layers.Rescaling(1 / 255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dropout(.5), layers.Dense(7, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(training.prefetch(tf.data.AUTOTUNE),
                        validation_data=validation.prefetch(tf.data.AUTOTUNE),
                        epochs=epochs)
    loss, accuracy = model.evaluate(validation, verbose=0)
    actual = np.concatenate([y.numpy() for _, y in validation])
    predicted = np.argmax(model.predict(validation, verbose=0), axis=1)
    result = {
        "validation_accuracy": float(accuracy),
        "validation_loss": float(loss),
        "class_names": names,
        "classification_report": classification_report(
            actual, predicted, target_names=names, output_dict=True,
            zero_division=0),
        "note": "Validation split is not an independent test set"
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    ConfusionMatrixDisplay.from_predictions(
        actual, predicted, display_labels=names, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=140)
    plt.close()
    for metric in ("accuracy", "loss"):
        plt.plot(history.history[metric], label="training")
        plt.plot(history.history["val_" + metric], label="validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out / (metric + "_curve.png"), dpi=140)
        plt.close()
    model.save(out / "model.keras")
    print(f"Validation accuracy: {accuracy:.4f}; loss: {loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size)

"""Train a VGG-style CNN on CIFAR-10 and save everything the report needs.

Outputs (in cnn/results/):
    cifar10_cnn.keras        trained model
    model_summary.txt        all layers + parameter counts   (brief output 1)
    results.json             test accuracy, loss, per-class accuracy (output 2)
    training_curves.png      accuracy / loss vs epoch
    confusion_matrix.png     10x10 confusion matrix (error analysis)

Run:  python train_cnn.py --epochs 60
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import CLASS_NAMES, MODEL_PATH, RESULTS_DIR


def build_model():
    """VGG-style CNN: 3 conv blocks (2x Conv-BN-ReLU, MaxPool, Dropout) + classifier."""

    def conv_block(x, filters, drop):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        return layers.Dropout(drop)(x)

    inputs = keras.Input(shape=(32, 32, 3), name="image_0_255")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)   # raw pixels in
    x = conv_block(x, 32, 0.2)
    x = conv_block(x, 64, 0.3)
    x = conv_block(x, 128, 0.4)
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax", name="class_probs")(x)
    return keras.Model(inputs, outputs, name="cifar10_vgg_style_cnn")


def make_datasets(batch_size):
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
    y_tr, y_te = y_tr.flatten(), y_te.flatten()
    # hold out 5,000 training images for validation (never touch the test set)
    x_val, y_val = x_tr[-5000:], y_tr[-5000:]
    x_tr, y_tr = x_tr[:-5000], y_tr[:-5000]

    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomRotation(0.05),
    ])
    train = (tf.data.Dataset.from_tensor_slices((x_tr.astype("float32"), y_tr))
             .shuffle(10_000).batch(batch_size)
             .map(lambda x, y: (augment(x, training=True), y),
                  num_parallel_calls=tf.data.AUTOTUNE)
             .prefetch(tf.data.AUTOTUNE))
    val = tf.data.Dataset.from_tensor_slices((x_val.astype("float32"), y_val)).batch(256)
    return train, val, (x_te.astype("float32"), y_te)


def plot_curves(history, path):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(h["accuracy"], label="train")
    ax[0].plot(h["val_accuracy"], label="validation")
    ax[0].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax[1].plot(h["loss"], label="train")
    ax[1].plot(h["val_loss"], label="validation")
    ax[1].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="predicted", ylabel="true",
           title="CIFAR-10 test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    model = build_model()
    model.summary()                                   # printed for the screenshot
    with open(RESULTS_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train, val, (x_te, y_te) = make_datasets(args.batch_size)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ]
    history = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)

    # ---- final evaluation on the untouched 10,000-image test set ----
    test_loss, test_acc = model.evaluate(x_te, y_te, batch_size=256, verbose=0)
    probs = model.predict(x_te, batch_size=256, verbose=0)
    pred = probs.argmax(axis=1)
    cm = tf.math.confusion_matrix(y_te, pred, num_classes=10).numpy()
    per_class = {CLASS_NAMES[i]: float(cm[i, i] / cm[i].sum()) for i in range(10)}

    model.save(MODEL_PATH)
    plot_curves(history, RESULTS_DIR / "training_curves.png")
    plot_confusion(cm, RESULTS_DIR / "confusion_matrix.png")

    # most confused (true -> predicted) pairs for the error analysis
    off = cm.copy()
    np.fill_diagonal(off, 0)
    worst = sorted(((int(off[i, j]), CLASS_NAMES[i], CLASS_NAMES[j])
                    for i in range(10) for j in range(10)), reverse=True)[:5]

    results = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_run": len(history.history["loss"]),
        "parameters": int(model.count_params()),
        "per_class_accuracy": per_class,
        "top_confusions": [{"count": c, "true": t, "predicted": p} for c, t, p in worst],
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nTEST ACCURACY: {test_acc * 100:.2f}%   (params: {model.count_params():,})")
    print("Per-class:", {k: f"{v * 100:.1f}%" for k, v in per_class.items()})
    print("Top confusions:", worst)


if __name__ == "__main__":
    main()

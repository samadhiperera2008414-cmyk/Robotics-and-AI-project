"""Optimisation experiment: shrink the CNN with post-training quantisation and
measure the accuracy / size / latency trade-off (supports the optimisation proposal).

Variants compared:
    keras_fp32      original model (TensorFlow runtime)
    tflite_fp32     TFLite, no quantisation
    tflite_dynamic  dynamic-range quantisation (int8 weights)
    tflite_fp16     float16 weights
    tflite_int8     full-integer quantisation (representative dataset)

Run (after train_cnn.py):  python optimise_model.py
Outputs: results/optimisation.json, results/optimisation_chart.png, *.tflite files
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from common import MODEL_PATH, RESULTS_DIR


def convert(model, name, x_rep):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if name != "tflite_fp32":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if name == "tflite_fp16":
        conv.target_spec.supported_types = [tf.float16]
    if name == "tflite_int8":
        def rep():
            for i in range(len(x_rep)):
                yield [x_rep[i:i + 1]]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # input/output stay float32 so the controller code does not change
    blob = conv.convert()
    path = RESULTS_DIR / f"{name}.tflite"
    path.write_bytes(blob)
    return path


def run_tflite(path, x, n_threads=1):
    interp = tf.lite.Interpreter(model_path=str(path), num_threads=n_threads)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds, times = [], []
    for img in x:
        interp.set_tensor(inp["index"], img[None])
        t0 = time.perf_counter()
        interp.invoke()
        times.append(time.perf_counter() - t0)
        preds.append(interp.get_tensor(out["index"])[0].argmax())
    return np.array(preds), np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=2000, help="test images used per variant")
    args = ap.parse_args()

    model = keras.models.load_model(MODEL_PATH)
    (x_tr, _), (x_te, y_te) = keras.datasets.cifar10.load_data()
    y_te = y_te.flatten()[:args.eval_n]
    x_te = x_te[:args.eval_n].astype("float32")
    x_rep = x_tr[:200].astype("float32")

    results = {}

    # --- baseline: Keras, batch size 1 (what the robot would really do) ---
    model.predict(x_te[:1], verbose=0)                     # warm-up
    times, preds = [], []
    for img in x_te[:300]:
        t0 = time.perf_counter()
        p = model.predict(img[None], verbose=0)
        times.append(time.perf_counter() - t0)
        preds.append(p.argmax())
    results["keras_fp32"] = {
        "accuracy": float((np.array(preds) == y_te[:300]).mean()),
        "size_kb": (MODEL_PATH.stat().st_size) / 1024,
        "latency_ms": float(np.median(times) * 1000),
    }

    # --- TFLite variants ---
    for name in ["tflite_fp32", "tflite_dynamic", "tflite_fp16", "tflite_int8"]:
        path = convert(model, name, x_rep)
        preds, times = run_tflite(path, x_te)
        results[name] = {
            "accuracy": float((preds == y_te).mean()),
            "size_kb": path.stat().st_size / 1024,
            "latency_ms": float(np.median(times) * 1000),
        }
        print(name, results[name])

    for r in results.values():
        r["fps_equivalent"] = 1000.0 / r["latency_ms"]
    (RESULTS_DIR / "optimisation.json").write_text(json.dumps(results, indent=2))

    # --- chart ---
    names = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, key, title in zip(ax, ["accuracy", "size_kb", "latency_ms"],
                             ["Accuracy", "Model size (KB)", "Median latency (ms)"]):
        vals = [results[n][key] * (100 if key == "accuracy" else 1) for n in names]
        a.bar(range(len(names)), vals, color="#2b7a78")
        a.set_xticks(range(len(names)))
        a.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        a.set_title(title)
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "optimisation_chart.png", dpi=200)
    print("Saved results/optimisation.json and optimisation_chart.png")


if __name__ == "__main__":
    main()

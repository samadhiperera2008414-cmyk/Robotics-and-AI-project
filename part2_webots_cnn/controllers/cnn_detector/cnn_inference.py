"""CNN inference back-ends for the Webots controller.

Three interchangeable classifiers, all exposing  predict(rgb_image) -> (label, confidence):

    LocalClassifier   loads a .tflite (fast, quantised) or .keras model in-process
    RemoteClassifier  sends a 32x32 crop to cnn/inference_server.py over TCP
                      (offloads TensorFlow out of Webots' Python - the "edge/offload" option)
    NullClassifier    used when no model is available; the robot falls back to
                      colour-only detection instead of crashing

Inputs are RGB uint8 crops of any size; they are centre-cropped and resized
to 32x32 and fed to the network as raw 0-255 floats (the model rescales itself).
"""
import json
import socket
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def to_cifar_input(rgb):
    h, w = rgb.shape[:2]
    s = min(h, w)
    rgb = rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")


class NullClassifier:
    name = "none"

    def predict(self, rgb):
        return "unclassified", 0.0


class LocalClassifier:
    """In-process inference. Prefers TFLite because it is much lighter than Keras."""

    def __init__(self, model_path):
        self.path = Path(model_path)
        self.name = f"local:{self.path.name}"
        if self.path.suffix == ".tflite":
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            self._interp = Interpreter(model_path=str(self.path), num_threads=2)
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]["index"]
            self._out = self._interp.get_output_details()[0]["index"]
            self._keras = None
        else:
            from tensorflow import keras
            self._keras = keras.models.load_model(str(self.path))

    def _probs(self, x):
        if self._keras is not None:
            return self._keras.predict(x[None], verbose=0)[0]
        self._interp.set_tensor(self._in, x[None])
        self._interp.invoke()
        return self._interp.get_tensor(self._out)[0]

    def predict(self, rgb):
        probs = self._probs(to_cifar_input(rgb))
        k = int(np.argmax(probs))
        return CLASS_NAMES[k], float(probs[k])


class RemoteClassifier:
    """Sends the 32x32 crop (3072 raw bytes) to inference_server.py, reads one JSON line."""

    def __init__(self, host="127.0.0.1", port=5005, timeout=1.0):
        self.name = f"remote:{host}:{port}"
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._file = self._sock.makefile("rb")

    def predict(self, rgb):
        payload = to_cifar_input(rgb).astype("uint8").tobytes()
        self._sock.sendall(struct.pack("!I", len(payload)) + payload)
        reply = json.loads(self._file.readline())
        return reply["label"], reply["confidence"]


def make_classifier(model_path=None, remote=None):
    """Build the best available classifier; never raises (robot must keep running)."""
    if remote:
        try:
            host, port = remote.split(":")
            return RemoteClassifier(host, int(port))
        except Exception as exc:                      # noqa: BLE001
            print(f"[cnn] remote classifier unavailable ({exc}); trying local model", file=sys.stderr)
    if model_path and Path(model_path).exists():
        try:
            return LocalClassifier(model_path)
        except Exception as exc:                      # noqa: BLE001
            print(f"[cnn] could not load model ({exc}); colour-only mode", file=sys.stderr)
    else:
        print(f"[cnn] model not found at {model_path}; colour-only mode", file=sys.stderr)
    return NullClassifier()

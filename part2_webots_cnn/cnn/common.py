"""Shared helpers for the CIFAR-10 CNN scripts.

The saved Keras model contains its own Rescaling(1/255) layer, so every
script (training, prediction, TFLite export, Webots controller) feeds the
network RAW pixel values in the range 0-255.  This avoids the classic
"trained on 0-1, deployed on 0-255" preprocessing mismatch.
"""
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
IMG_SIZE = 32

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODEL_PATH = RESULTS_DIR / "cifar10_cnn.keras"


def center_square_crop(img):
    """Crop the largest centred square from an HxWx3 image."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def to_cifar_input(rgb, crop=True):
    """RGB uint8 image (any size) -> 32x32x3 float32 array in range 0-255."""
    if crop:
        rgb = center_square_crop(rgb)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype("float32")


def load_image_rgb(path):
    """Read an image file and return it as an RGB uint8 array."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

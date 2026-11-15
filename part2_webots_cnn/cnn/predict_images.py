"""Show the trained CNN's predictions on your own images (brief output 3).

Put at least three photos (jpg/png) of CIFAR-10 objects in cnn/my_images/ and run:
    python predict_images.py
    python predict_images.py --images some_folder --model results/cifar10_cnn.keras

Saves results/own_image_predictions.png (insert into report + slides).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from common import (CLASS_NAMES, MODEL_PATH, RESULTS_DIR, ROOT,
                    load_image_rgb, to_cifar_input)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "my_images"))
    ap.add_argument("--model", default=str(MODEL_PATH))
    args = ap.parse_args()

    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS)
    if len(files) < 3:
        raise SystemExit(f"Found {len(files)} image(s) in {args.images}; need at least 3.")

    model = keras.models.load_model(args.model)
    fig, axes = plt.subplots(len(files), 2, figsize=(7.5, 2.6 * len(files)))
    axes = np.atleast_2d(axes)

    for row, path in enumerate(files):
        rgb = load_image_rgb(path)
        x = to_cifar_input(rgb)
        probs = model.predict(x[None], verbose=0)[0]
        top3 = probs.argsort()[::-1][:3]
        print(f"\n{path.name}")
        for k in top3:
            print(f"   {CLASS_NAMES[k]:<11s} {probs[k] * 100:5.1f}%")

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(path.name, fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].barh([CLASS_NAMES[k] for k in top3][::-1],
                          [probs[k] for k in top3][::-1], color="#2b7a78")
        axes[row, 1].set(xlim=(0, 1), title=f"Prediction: {CLASS_NAMES[top3[0]]} "
                                            f"({probs[top3[0]] * 100:.1f}%)")
    fig.tight_layout()
    out = RESULTS_DIR / "own_image_predictions.png"
    fig.savefig(out, dpi=200)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

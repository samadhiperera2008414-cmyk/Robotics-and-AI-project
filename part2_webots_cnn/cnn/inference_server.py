"""Optional inference server - runs the CNN OUTSIDE Webots (offloading).

Why: Webots uses its own Python interpreter, which often lacks TensorFlow, and heavy
inference inside the control loop can stall the robot.  Start this server in your normal
Python environment, then launch the controller with  CNN_REMOTE=127.0.0.1:5005.

Protocol: client sends 4-byte big-endian length + 3072 raw bytes (32x32x3 RGB uint8);
server answers with one JSON line {"label": ..., "confidence": ...}.

    python inference_server.py --model results/tflite_dynamic.tflite
"""
import argparse
import json
import socketserver
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "controllers" / "cnn_detector"))
from cnn_inference import CLASS_NAMES, LocalClassifier  # noqa: E402


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        clf = self.server.clf
        while True:
            header = self.rfile.read(4)
            if len(header) < 4:
                return
            (n,) = struct.unpack("!I", header)
            raw = self.rfile.read(n)
            if len(raw) < n:
                return
            img = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 3)
            probs = clf._probs(img.astype("float32"))
            k = int(np.argmax(probs))
            self.wfile.write((json.dumps({"label": CLASS_NAMES[k],
                                          "confidence": float(probs[k])}) + "\n").encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/tflite_dynamic.tflite")
    ap.add_argument("--port", type=int, default=5005)
    args = ap.parse_args()
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), Handler) as srv:
        srv.clf = LocalClassifier(args.model)
        print(f"Serving {args.model} on 127.0.0.1:{args.port}  (Ctrl+C to stop)")
        srv.serve_forever()


if __name__ == "__main__":
    main()

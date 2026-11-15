"""cnn_detector - Webots e-puck controller: colour pre-filter + CIFAR-10 CNN.

What the robot does (finite state machine):

    SEARCH   -> wander and avoid walls until a coloured object appears in the camera
    ALIGN    -> turn until the object is centred in the image
    APPROACH -> drive towards it until it fills enough of the image
    CLASSIFY -> crop the object from the camera frame and run the CNN on it
    RECOVER  -> reverse and turn away so the next object can be found

Optimisations so the CNN does not disturb robot motion (see report, Section 3.6):
    1. Camera sampling period = TIME_STEP * PROCESS_EVERY_N (frame skipping).
    2. Cheap HSV colour gate: the CNN runs only on a stable, large-enough blob.
    3. Only a 32x32 crop is classified, never the full frame.
    4. Quantised TFLite model, or optional offloading to cnn/inference_server.py.
    5. Per-object cooldown, so one object is not classified repeatedly.

Keyboard (click the 3D view first):
    C = classify the biggest coloured object / image centre right now
    P = save the current camera frame to snapshots/
    S = print the detection summary

Environment variables (all optional):
    CNN_MODEL   path to .tflite / .keras   (default: cnn/results/tflite_dynamic.tflite)
    CNN_REMOTE  host:port of inference_server.py, e.g. 127.0.0.1:5005
    RUN_SECONDS stop searching after this many simulated seconds (default 180)
"""
import os
import time
from collections import Counter
from enum import Enum, auto
from pathlib import Path

import cv2
from controller import Robot                    # Webots Python API

from cnn_inference import make_classifier
from colour_utils import bgr_from_webots, crop_blob, find_blobs

# ------------------------------------------------------------------ settings
TIME_STEP = 64                    # ms, multiple of the world's basicTimeStep
MAX_SPEED = 6.28                  # e-puck wheel limit (rad/s)
CRUISE = 0.6 * MAX_SPEED
PROCESS_EVERY_N = 2               # vision runs every N steps (frame skipping)
OBSTACLE_THRESHOLD = 90.0         # proximity value that counts as "wall ahead"
APPROACH_WIDTH_FRACTION = 0.45    # stop when the blob is this wide (fraction of image)
ALIGN_TOLERANCE = 0.12            # |normalised x error| below which we are centred
CONFIRM_FRAMES = 2                # blob must be seen this many processed frames in a row
LOST_FRAMES = 8                   # blob missing this long -> back to SEARCH
IGNORE_STEPS = 300                # cooldown for a colour after it was reported
MIN_CONFIDENCE = 0.40             # below this the CNN answer is reported as "uncertain"

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent.parent / "cnn" / "results" / "tflite_dynamic.tflite"
SNAP_DIR = HERE / "snapshots"


class State(Enum):
    SEARCH = auto()
    ALIGN = auto()
    APPROACH = auto()
    CLASSIFY = auto()
    RECOVER = auto()


# ------------------------------------------------------------------ helpers
def clamp(v, lo=-MAX_SPEED, hi=MAX_SPEED):
    return max(lo, min(hi, v))


class EpuckDetector:
    def __init__(self):
        self.robot = Robot()
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(TIME_STEP * PROCESS_EVERY_N)          # optimisation 1
        self.width, self.height = self.camera.getWidth(), self.camera.getHeight()

        self.left = self.robot.getDevice("left wheel motor")
        self.right = self.robot.getDevice("right wheel motor")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))                         # velocity control
            m.setVelocity(0.0)

        self.ps = []
        for i in range(8):
            s = self.robot.getDevice(f"ps{i}")
            s.enable(TIME_STEP)
            self.ps.append(s)

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(TIME_STEP)

        model = os.environ.get("CNN_MODEL", str(DEFAULT_MODEL))
        self.classifier = make_classifier(model, os.environ.get("CNN_REMOTE"))
        self.run_seconds = float(os.environ.get("RUN_SECONDS", 180))

        self.state = State.SEARCH
        self.step_count = 0
        self.seen_streak = 0
        self.lost_streak = 0
        self.target = None                  # most recent Blob being tracked
        self.frame = None                   # most recent BGR frame
        self.recover_steps = 0
        self.ignore_until = {}              # colour -> step when cooldown ends
        self.detections = []                # list of dicts, used for the summary
        self.infer_times = []
        self.vision_times = []
        self.last_turn_dir = 1

    # -------------------------------------------------------------- sensing
    def front_obstacle(self):
        """True if a wall/object is close in front (ps0, ps1, ps6, ps7)."""
        vals = [self.ps[i].getValue() for i in (0, 1, 6, 7)]
        return max(vals) > OBSTACLE_THRESHOLD

    def grab_frame(self):
        raw = self.camera.getImage()
        if raw is None:                                         # camera not ready yet
            return None
        return bgr_from_webots(raw, self.width, self.height)

    def best_blob(self, frame):
        """Largest blob whose colour is not on cooldown (colour pre-filter)."""
        for blob in find_blobs(frame):
            if self.ignore_until.get(blob.colour, 0) <= self.step_count:
                return blob
        return None

    # -------------------------------------------------------------- actions
    def drive(self, left, right):
        self.left.setVelocity(clamp(left))
        self.right.setVelocity(clamp(right))

    def classify(self, blob, frame):
        """Run the CNN on the blob crop and store the result."""
        crop = crop_blob(frame, blob)
        t0 = time.perf_counter()
        try:
            label, conf = self.classifier.predict(crop)
        except Exception as exc:                                # noqa: BLE001 - keep robot alive
            print(f"[cnn] inference failed: {exc}")
            label, conf = "error", 0.0
        self.infer_times.append((time.perf_counter() - t0) * 1000)

        if conf < MIN_CONFIDENCE and label not in ("unclassified", "error"):
            label = f"uncertain ({label}?)"
        record = {"t": round(self.robot.getTime(), 1), "colour": blob.colour,
                  "class": label, "confidence": conf}
        self.detections.append(record)
        print(f"[detect] t={record['t']:>6}s  colour={blob.colour:<6}  "
              f"CNN={label:<22}  conf={conf * 100:5.1f}%  ({self.infer_times[-1]:.1f} ms)")
        self.save_snapshot(frame, tag=f"{blob.colour}_{label.split()[0]}")
        return record

    def save_snapshot(self, frame, tag="frame"):
        SNAP_DIR.mkdir(exist_ok=True)
        path = SNAP_DIR / f"{tag}_{int(self.robot.getTime() * 1000)}.png"
        cv2.imwrite(str(path), cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        return path

    def summary(self):
        print("\n" + "=" * 62)
        print("DETECTION SUMMARY")
        print("=" * 62)
        if not self.detections:
            print("No objects were classified.")
        for i, d in enumerate(self.detections, 1):
            print(f"{i:>2}. {d['colour']:<7} -> {d['class']:<22} {d['confidence'] * 100:5.1f}%")
        counts = Counter(d["colour"] for d in self.detections)
        print("By colour:", dict(counts))
        if self.infer_times:
            avg = sum(self.infer_times) / len(self.infer_times)
            print(f"CNN backend: {self.classifier.name}   mean inference: {avg:.1f} ms")
        if self.vision_times:
            avg_v = sum(self.vision_times) / len(self.vision_times)
            print(f"Colour-filter vision: {avg_v:.2f} ms/frame (every {PROCESS_EVERY_N} steps)")
        print("=" * 62)

    # -------------------------------------------------------------- keyboard
    def handle_keys(self, frame):
        key = self.keyboard.getKey()
        while key != -1:
            if key in (ord("C"), ord("c")) and frame is not None:
                blob = (find_blobs(frame) or [None])[0]
                if blob is None:                                  # fall back to the image centre
                    h, w = frame.shape[:2]
                    s = min(h, w) // 2
                    crop = cv2.cvtColor(frame[h // 2 - s // 2:h // 2 + s // 2,
                                              w // 2 - s // 2:w // 2 + s // 2], cv2.COLOR_BGR2RGB)
                    label, conf = self.classifier.predict(crop)
                    print(f"[manual] centre crop -> {label} ({conf * 100:.1f}%)")
                else:
                    self.classify(blob, frame)
            elif key in (ord("P"), ord("p")) and frame is not None:
                print("[manual] saved", self.save_snapshot(frame, "manual"))
            elif key in (ord("S"), ord("s")):
                self.summary()
            key = self.keyboard.getKey()

    # -------------------------------------------------------------- state machine
    def update_state(self, frame, blob):
        """Advance the FSM. `blob` is None when nothing (new) is visible."""
        err = 0.0
        if blob is not None:
            self.target = blob
            err = (blob.cx - self.width / 2) / (self.width / 2)    # -1 (left) .. +1 (right)
            self.seen_streak += 1
            self.lost_streak = 0
        else:
            self.seen_streak = 0
            self.lost_streak += 1

        if self.state == State.SEARCH:
            if blob is not None and self.seen_streak >= CONFIRM_FRAMES:   # optimisation 2
                self.state = State.ALIGN
        elif self.state == State.ALIGN:
            if self.lost_streak > LOST_FRAMES:
                self.state = State.SEARCH
            elif blob is not None and abs(err) < ALIGN_TOLERANCE:
                self.state = State.APPROACH
        elif self.state == State.APPROACH:
            if self.lost_streak > LOST_FRAMES:
                self.state = State.SEARCH
            elif blob is not None and (blob.w >= APPROACH_WIDTH_FRACTION * self.width
                                       or self.front_obstacle()):
                self.state = State.CLASSIFY
        return err

    def act(self, err, frame):
        if self.state == State.SEARCH:
            if self.front_obstacle():
                self.drive(-CRUISE * 0.2, CRUISE)                  # turn away from the wall
            else:
                self.drive(CRUISE * 0.9, CRUISE)                   # gentle arc covers the arena
        elif self.state == State.ALIGN:
            turn = clamp(3.0 * err, -2.5, 2.5)
            self.drive(turn, -turn)
            self.last_turn_dir = 1 if err >= 0 else -1
        elif self.state == State.APPROACH:
            turn = 2.0 * err
            self.drive(CRUISE - turn, CRUISE + turn)
        elif self.state == State.CLASSIFY:
            self.drive(0.0, 0.0)
            if frame is not None and self.target is not None:
                rec = self.classify(self.target, frame)
                self.ignore_until[rec["colour"]] = self.step_count + IGNORE_STEPS
            self.state = State.RECOVER
            self.recover_steps = 35
        elif self.state == State.RECOVER:
            self.recover_steps -= 1
            if self.recover_steps > 25:
                self.drive(-CRUISE, -CRUISE)                       # back off
            else:
                self.drive(CRUISE, -CRUISE)                        # spin away
            if self.recover_steps <= 0:
                self.state = State.SEARCH
                self.seen_streak = 0

    # -------------------------------------------------------------- main loop
    def run(self):
        print(f"[cnn_detector] camera {self.width}x{self.height}, classifier = {self.classifier.name}")
        err = 0.0
        try:
            while self.robot.step(TIME_STEP) != -1:
                self.step_count += 1
                if self.robot.getTime() > self.run_seconds:
                    print("[cnn_detector] time limit reached")
                    break

                if self.step_count % PROCESS_EVERY_N == 0:         # optimisation 1: frame skipping
                    frame = self.grab_frame()
                    if frame is not None:
                        self.frame = frame
                        t0 = time.perf_counter()
                        blob = self.best_blob(frame)
                        self.vision_times.append((time.perf_counter() - t0) * 1000)
                        err = self.update_state(frame, blob)
                self.handle_keys(self.frame)
                self.act(err, self.frame)
        finally:
            self.drive(0.0, 0.0)
            self.summary()


if __name__ == "__main__":
    EpuckDetector().run()

"""HSV colour-blob detection used as a cheap pre-filter in front of the CNN.

OpenCV HSV ranges: H 0-179, S 0-255, V 0-255.  Red wraps around H=0, so it needs two ranges.
Tune these ranges for your own Webots lighting; they are deliberately kept in one table.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# colour name -> list of (lower HSV, upper HSV) ranges
HSV_RANGES = {
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green":  [((40, 80, 50), (85, 255, 255))],
    "blue":   [((95, 100, 50), (130, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
}
MIN_AREA_FRACTION = 0.004          # ignore blobs smaller than 0.4 % of the image
_KERNEL = np.ones((3, 3), np.uint8)


@dataclass
class Blob:
    colour: str
    x: int            # bounding box (pixels)
    y: int
    w: int
    h: int
    area: float       # contour area (pixels^2)

    @property
    def cx(self):
        return self.x + self.w / 2.0

    @property
    def cy(self):
        return self.y + self.h / 2.0


def bgr_from_webots(raw, width, height):
    """Webots Camera.getImage() bytes (BGRA) -> BGR numpy array (H, W, 3)."""
    return np.frombuffer(raw, np.uint8).reshape((height, width, 4))[:, :, :3].copy()


def find_blobs(bgr, colours=None):
    """Return all colour blobs larger than MIN_AREA_FRACTION, biggest first."""
    colours = colours or HSV_RANGES.keys()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    min_area = MIN_AREA_FRACTION * bgr.shape[0] * bgr.shape[1]
    blobs = []
    for name in colours:
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in HSV_RANGES[name]:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)        # remove speckle
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                blobs.append(Blob(name, x, y, w, h, area))
    return sorted(blobs, key=lambda b: b.area, reverse=True)


def crop_blob(bgr, blob, pad=0.15):
    """Crop a padded square around the blob and return it as RGB for the CNN."""
    side = int(max(blob.w, blob.h) * (1 + 2 * pad))
    half = side // 2
    cx, cy = int(blob.cx), int(blob.cy)
    y0, y1 = max(0, cy - half), min(bgr.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(bgr.shape[1], cx + half)
    return cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

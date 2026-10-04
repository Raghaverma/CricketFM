"""Burned-in app overlays in the footage: frames that already carry another model's pose.

Many STANCE_ANALYSIS uploads are recordings of the app's live view: a skeleton (white lines, pure
red dots on the joints) and coaching text boxes ("Wrong Pose, Please Stand Still", each with a red
dot in its corner) are drawn into the pixels. Such a frame can't be a pose test: an annotator would
copy the drawn joints, and a model would be shown them. The same goes for a shot clip shown to a
VLM, which would get a pose overlay the protocol reserves for L5.

The overlay's red is pure (BGR about 8, 0, 246, measured), far more saturated than shirts, balls
or stumps, and its dots are small and round. dots() counts such dots in a frame; a frame with at
least MIN_DOTS is an overlay frame.
"""
from __future__ import annotations

import cv2
import numpy as np

WIDTH = 640                     # frames are scaled to this width first, so the limits below are resolution-free
MIN_AREA, MAX_AREA = 6, 400     # px at WIDTH; a joint dot is about 45-80 px there, a text-box dot about 16
MIN_FILL = 0.45                 # dot area / its bounding box (a filled circle is 0.79)
MIN_DOTS = 3


def dots(frame: np.ndarray) -> int:
    """Number of overlay-red, dot-shaped blobs in a BGR frame."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    b, g, r = (small[..., k].astype(np.int16) for k in range(3))
    red = (r >= 190) & (g <= 70) & (b <= 80) & (r - np.maximum(g, b) >= 150)
    n, _, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
    count = 0
    for i in range(1, n):
        bw, bh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA <= area <= MAX_AREA and 0.5 <= bw / bh <= 2 and area / (bw * bh) >= MIN_FILL:
            count += 1
    return count

"""Strict parsing for the public aspect-ratio option, and the exact-ratio fit of a delivered box."""

import math


def parse_aspect(value):
    """Parse a positive finite ratio written as W:H or as one float."""
    try:
        if value.count(":") == 1:
            width, height = (float(part) for part in value.split(":"))
            if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
                raise ValueError
            return width / height
        if ":" not in value:
            ratio = float(value)
            if math.isfinite(ratio) and ratio > 0:
                return ratio
    except (TypeError, ValueError, OverflowError):
        pass
    raise ValueError("aspect must be W:H (two positive finite numbers) or one positive finite number")


def fit_aspect(box, ratio, W, H):
    """Return ``box`` shrunk to exactly ``ratio`` (width / height) inside a W x H image, keeping its
    centre and its area where the image allows; a box already at the ratio comes back unchanged."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w0, h0 = x2 - x1, y2 - y1
    if h0 <= 0 or abs(w0 / h0 - ratio) <= 1e-3 * ratio:
        return tuple(box)
    w = min(math.sqrt(w0 * h0 * ratio), float(W))
    h = w / ratio
    if h > H:
        h = float(H); w = h * ratio
    nx1 = min(max((x1 + x2) / 2 - w / 2, 0.0), W - w)
    ny1 = min(max((y1 + y2) / 2 - h / 2, 0.0), H - h)
    return (nx1, ny1, nx1 + w, ny1 + h)

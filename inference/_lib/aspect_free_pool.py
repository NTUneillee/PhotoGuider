"""Aspect enumeration for the free-form (aspect-free) pool, stage 1.

Constants + enum_aspects used by infer.py's free-form pool generator; the per-crop features of
that pool are computed by consultant_features.stage_c_features (the same loop the twin rows use).
Values are byte-identical extracts of the validated benchmark chain."""


SCALES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


POSITIONS = [(cx, cy) for cx in [0.15, 0.30, 0.45, 0.55, 0.70, 0.85]
                      for cy in [0.15, 0.30, 0.45, 0.55, 0.70, 0.85]]


ASPECTS_FIXED = [16/9, 3/2, 4/3, 1/1, 3/4, 2/3, 9/16]


ASPECT_DEDUP_TOL = 0.02


def enum_aspects(W, H):
    """Fixed 7 aspects + original W/H, dedupe near-duplicates within 2%."""
    out = list(ASPECTS_FIXED)
    orig = W / H
    if all(abs(orig - a) / a > ASPECT_DEDUP_TOL for a in out):
        out.append(orig)
    return out

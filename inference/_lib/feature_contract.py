"""CPU-only candidate and extractor-output contracts."""
import numpy as np
from PIL import Image

KEYS = ["photo_id", "dpp_rank"]

def whole_view_rows(rows):
    result = []
    for r in rows:
        with Image.open(r["path"]) as image:
            w, h = image.size
        result.append(dict(photo_id=r["vid"], image_path=r["path"], src="view",
                           x1=0., y1=0., x2=float(w), y2=float(h),
                           img_w=float(w), img_h=float(h), gt_aspect=w/h, dpp_rank=0))
    return result

def validate_boxes(frame):
    for path, rows in frame.groupby("image_path"):
        with Image.open(path) as image:      # header only; the extractors decode the pixels
            w, h = image.size
        for r in rows.itertuples():
            box = np.array([r.x1, r.y1, r.x2, r.y2], dtype=float)
            if not (np.isfinite(box).all() and r.img_w == w and r.img_h == h
                    and 0 <= r.x1 < r.x2 <= w and 0 <= r.y1 < r.y2 <= h):
                raise AssertionError(f"candidate outside image or wrong dimensions: {path}, box={box}, image={w}x{h}")

def validate_rows(expected, actual, name, keys=KEYS):
    for label, frame in (("expected", expected), ("actual", actual)):
        if any(k not in frame for k in keys) or frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name}: missing/null/duplicate {label} candidate keys {keys}")
    wanted = set(expected[keys].itertuples(index=False, name=None))
    got = set(actual[keys].itertuples(index=False, name=None))
    if wanted != got:
        raise ValueError(f"{name}: candidate coverage mismatch: missing={len(wanted-got)}, unexpected={len(got-wanted)}")
    # Extractors may echo metadata columns (image_path, src, ...) alongside their features;
    # the contract applies to the numeric feature columns, of which at least one must exist.
    payload = actual.drop(columns=keys)
    if payload.shape[1] == 0:          # key-only frame: a pure coverage check
        return
    values = payload.select_dtypes(include=[np.number])
    if values.shape[1] == 0:
        raise ValueError(f"{name}: no numeric feature columns in extractor output")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: non-finite feature values")

"""Tiny dependency-free checks for the public aspect parser."""

import importlib.util
from pathlib import Path


PARSER = Path(__file__).resolve().parents[1] / "inference" / "_lib" / "aspect.py"
spec = importlib.util.spec_from_file_location("aspect_parser", PARSER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.parse_aspect("16:9") == 16 / 9
assert module.parse_aspect("1.5") == 1.5

for invalid in ("0", "-1", "1:0", "1:2:3", "nan", "inf", "1+1"):
    try:
        module.parse_aspect(invalid)
    except ValueError as exc:
        assert "positive finite" in str(exc)
    else:
        raise AssertionError(f"accepted invalid aspect: {invalid!r}")

fit = module.fit_aspect
r = 16 / 9
box = fit((0, 0, 1000, 900), r, 1000, 1000)                       # border-clipped protocol box
assert abs((box[2] - box[0]) / (box[3] - box[1]) - r) < 1e-9 and box[2] - box[0] == 1000 and 0 <= box[1] and box[3] <= 1000
assert fit((100, 100, 740, 460), r, 1000, 1000) == (100, 100, 740, 460)   # already 16:9: untouched
tall = fit((0, 0, 1000, 1000), 9 / 16, 1000, 1000)                 # ratio limited by image height
assert abs((tall[2] - tall[0]) / (tall[3] - tall[1]) - 9 / 16) < 1e-9 and tall[3] - tall[1] == 1000
edge = fit((900, 0, 1000, 300), 2.0, 1000, 1000)                   # centre near the border: shifted inside
assert 0 <= edge[0] and edge[2] <= 1000 and abs((edge[2] - edge[0]) / (edge[3] - edge[1]) - 2.0) < 1e-9
print("fit_aspect: PASS")

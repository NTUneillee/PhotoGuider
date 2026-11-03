#!/usr/bin/env python3
"""User-image input: EXIF orientation is applied, plain images pass through, unreadable files fail cleanly."""
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "_lib"))
from image_io import open_upright  # noqa: E402


def main():
    d = Path(tempfile.mkdtemp())
    up = Image.new("RGB", (40, 20), (255, 0, 0))
    up.paste((0, 0, 255), (0, 0, 10, 20))                 # upright view: blue strip on the left
    for orient, stored in [(6, up.transpose(Image.ROTATE_90)), (8, up.transpose(Image.ROTATE_270)),
                           (3, up.transpose(Image.ROTATE_180))]:
        ex = stored.getexif(); ex[0x0112] = orient
        stored.save(d / f"o{orient}.png", exif=ex)          # lossless so pixel checks are exact
        img, rotated = open_upright(d / f"o{orient}.png")
        assert rotated and img.size == (40, 20), (orient, img.size)
        assert img.getpixel((2, 10)) == (0, 0, 255) and img.getpixel((30, 10)) == (255, 0, 0), orient
    up.save(d / "plain.png")
    img, rotated = open_upright(d / "plain.png")
    assert not rotated and img.size == (40, 20)
    (d / "bad.jpg").write_text("not an image")
    try:
        open_upright(d / "bad.jpg")
    except SystemExit as e:
        assert "cannot read image" in str(e)
    else:
        raise AssertionError("unreadable file must exit")
    print("open_upright: EXIF 3/6/8 upright, plain passthrough, unreadable exits: PASS")


if __name__ == "__main__":
    main()

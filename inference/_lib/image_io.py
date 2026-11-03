"""Reading user images the way a viewer sees them."""
from PIL import Image, ImageOps

EXIF_ORIENTATION = 0x0112


def open_upright(image):
    """Open ``image`` with its EXIF orientation applied (phone and camera photos are often stored
    sideways). Returns (PIL image, rotated) where ``rotated`` says the stored pixels were re-oriented.
    Unreadable files exit with a message naming the supported formats."""
    try:
        raw = Image.open(image)
        raw.load()
    except Exception as exc:
        raise SystemExit(f"cannot read image {image}: {exc}. Supported: formats Pillow opens "
                         "(JPEG, PNG, WebP, TIFF, BMP, ...).") from exc
    rotated = raw.getexif().get(EXIF_ORIENTATION, 1) not in (None, 1)
    return (ImageOps.exif_transpose(raw) if rotated else raw), rotated

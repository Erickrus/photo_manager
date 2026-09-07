from ocrmac.ocrmac import livetext_from_image
from PIL import Image, ImageOps

# Apple's VisionKit ImageAnalyzer rejects any image whose width or height is
# >= 8192 (the limit is exclusive: 8192 itself fails). We cap strictly below
# that. Note the crash it raises is an ObjC exception that aborts the whole
# process and CANNOT be caught with try/except — so we must downscale before
# ever calling LiveText.
MAX_DIMENSION = 8000


def _load_image(image_path):
    """Open an image, fix EXIF rotation, and downscale if either side reaches
    the LiveText max dimension. Returns a PIL.Image."""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # honor camera orientation

    w, h = img.size
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(w, h)
        # floor via int(), then guard against rounding landing on the cap
        new_w = min(int(w * scale), MAX_DIMENSION)
        new_h = min(int(h * scale), MAX_DIMENSION)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def reconstruct_text(annotations, y_tolerance=0.01):
    """Order LiveText line-level results into reading order.

    Each annotation is a full line (text, conf, bbox) from
    livetext_from_image(..., unit='line'). LiveText already produces correct
    intra-line spacing (words spaced, CJK joined, abbreviations intact), so we
    only order the lines top-to-bottom, then left-to-right within a row.

    y_tolerance: lines whose vertical centers fall within this fraction of the
    image height are treated as the same visual row.
    """
    if not annotations:
        return ""

    items = []
    for text, conf, bbox in annotations:
        x, y, w, h = bbox
        items.append({"text": text, "x": x, "center_y": y + h / 2})

    # Top to bottom (larger center_y is higher in the image).
    items.sort(key=lambda i: -i["center_y"])

    lines = []
    current_row = [items[0]]
    for item in items[1:]:
        avg_y = sum(i["center_y"] for i in current_row) / len(current_row)
        if abs(item["center_y"] - avg_y) <= y_tolerance:
            current_row.append(item)
        else:
            current_row.sort(key=lambda i: i["x"])
            lines.append(" ".join(i["text"] for i in current_row))
            current_row = [item]

    current_row.sort(key=lambda i: i["x"])
    lines.append(" ".join(i["text"] for i in current_row))

    return "\n".join(lines)


def extract_text(image_path, y_tolerance=0.01):
    """Extract plain text from an image using the macOS native Vision
    framework (Apple LiveText) via ocrmac. Large images are downscaled to fit
    LiveText's max dimension. Returns reconstructed text."""
    img = _load_image(image_path)
    annotations = livetext_from_image(img, unit='line')
    return reconstruct_text(annotations, y_tolerance)


if __name__ == "__main__":
    import sys
    print(extract_text(sys.argv[1]))

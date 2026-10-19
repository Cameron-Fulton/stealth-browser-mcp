"""Fit browser screenshots into MCP image limits."""

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

INLINE_MAX_BYTES = 3_750_000
MAX_IMAGE_SIDE = 8000
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
MAX_DECODE_PIXELS = 100_000_000
DEFAULT_QUALITY = 85
FIT_ATTEMPTS = 6


def validate_options(format: str, quality: int, max_width: Optional[int]) -> str:
    """
    Normalize the format and check screenshot options before capturing.

    Args:
        format (str): png, jpeg, or jpg.
        quality (int): JPEG quality from 1 to 100.
        max_width (Optional[int]): Largest width to return, or None.

    Returns:
        str: Normalized format, png or jpeg.

    Raises:
        ValueError: When an option is out of range.
    """
    normalized = (format or "png").lower()
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized not in ("png", "jpeg"):
        raise ValueError(f"Unsupported screenshot format: {format}. Use 'png' or 'jpeg'.")
    if not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError(f"quality must be an integer from 1 to 100, got {quality!r}")
    if max_width is not None and (not isinstance(max_width, int) or max_width < 1):
        raise ValueError(f"max_width must be a positive integer, got {max_width!r}")
    return normalized


def _encode(image: Image.Image, format: str, quality: int) -> bytes:
    """
    Encode an image.

    Args:
        image (Image.Image): Image to encode.
        format (str): png or jpeg.
        quality (int): JPEG quality.

    Returns:
        bytes: Encoded image.
    """
    buffer = io.BytesIO()
    if format == "jpeg":
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _flatten_for_jpeg(image: Image.Image) -> Image.Image:
    """
    Composite transparency onto white, since JPEG has no alpha channel.

    Args:
        image (Image.Image): Decoded capture.

    Returns:
        Image.Image: RGB image.
    """
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB") if image.mode != "RGB" else image


def fit_screenshot(
    capture_path: Path,
    format: str,
    quality: int = DEFAULT_QUALITY,
    max_width: Optional[int] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Encode a captured screenshot so it fits inline MCP image limits.

    The image is scaled down, keeping its aspect ratio, when it is wider than
    max_width, longer than MAX_IMAGE_SIDE on either side, or larger than
    INLINE_MAX_BYTES once encoded. INLINE_MAX_BYTES keeps the base64 payload
    near 5 MB, which is what Claude and similar clients accept.

    Args:
        capture_path (Path): File written by the browser capture.
        format (str): png or jpeg.
        quality (int): JPEG quality from 1 to 100, ignored for PNG.
        max_width (Optional[int]): Largest width to return, or None.

    Returns:
        Tuple[bytes, Dict[str, Any]]: Encoded image and metadata with format,
        width, height, original_width, original_height, scale, quality, bytes,
        sha256, and complete.

    Raises:
        ValueError: When the capture is too large to decode safely or cannot be
        reduced below the inline limit.
    """
    if capture_path.stat().st_size > MAX_CAPTURE_BYTES:
        raise ValueError("Screenshot exceeds inline capture limit; use file_path for export.")
    with Image.open(capture_path) as captured:
        original_width, original_height = captured.size
        if original_width * original_height > MAX_DECODE_PIXELS:
            raise ValueError("Screenshot dimensions exceed the decode limit; use file_path for export.")
        decoded = captured.copy()
    image = _flatten_for_jpeg(decoded) if format == "jpeg" else decoded

    scale = 1.0
    if max_width and max_width < original_width:
        scale = max_width / original_width
    scale = min(scale, MAX_IMAGE_SIDE / max(original_width, original_height))

    for _ in range(FIT_ATTEMPTS):
        size = (max(1, round(original_width * scale)), max(1, round(original_height * scale)))
        resized = image if size == image.size else image.resize(size, Image.LANCZOS)
        data = _encode(resized, format, quality)
        if len(data) <= INLINE_MAX_BYTES:
            return data, {
                "format": format,
                "width": size[0],
                "height": size[1],
                "original_width": original_width,
                "original_height": original_height,
                "scale": round(size[0] / original_width, 4),
                "quality": quality if format == "jpeg" else None,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "complete": True,
            }
        shrink = (INLINE_MAX_BYTES / len(data)) ** 0.5 * 0.95
        scale *= min(0.9, max(0.5, shrink))
    raise ValueError(
        "Screenshot could not be reduced below the inline limit; lower quality, set max_width, or use file_path."
    )

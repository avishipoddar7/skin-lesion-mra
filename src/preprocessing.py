"""Stage 1: image loading/validation, square padding, and morphological hair removal."""
import cv2
import numpy as np

import config
from src.errors import InvalidImageError


def decode_image(data: bytes, filename: str = "") -> np.ndarray:
    """Validate and decode uploaded bytes into an RGB uint8 array."""
    if filename and not filename.lower().endswith(config.ALLOWED_EXT):
        raise InvalidImageError(
            "Unsupported file type. Supported formats: " + ", ".join(config.ALLOWED_EXT))
    if len(data) > config.MAX_BYTES:
        raise InvalidImageError("File is larger than 10 MB.")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise InvalidImageError("The file could not be read as an image.")
    if min(bgr.shape[:2]) < config.MIN_SIDE:
        raise InvalidImageError(
            f"Image is too small (minimum {config.MIN_SIDE}x{config.MIN_SIDE} pixels).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pad_square(img, value):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=value)


def prepare(rgb: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Pad to a square with the image's median colour (keeps lesion shape, and the flat
    padding triggers neither hair detection nor Otsu) and resize to size x size."""
    sq = _pad_square(rgb, [int(v) for v in np.median(rgb.reshape(-1, 3), axis=0)])
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def prepare_mask(mask: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Same geometry as prepare(), for expert masks (zero padding, nearest-neighbour)."""
    sq = _pad_square(mask.astype(np.uint8), 0)
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def remove_hair(rgb, kernel_size=config.HAIR_KERNEL, threshold=config.HAIR_THRESHOLD):
    """Black-hat -> threshold -> drop small specks -> dilate -> Telea inpainting.
    Returns (clean_rgb, hair_mask)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hair, connectivity=8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= config.HAIR_MIN_AREA]
    hair = np.isin(labels, big).astype(np.uint8) * 255
    hair = cv2.dilate(hair, np.ones((config.HAIR_DILATE, config.HAIR_DILATE), np.uint8))
    clean = cv2.inpaint(rgb, hair, config.INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return clean, hair > 0

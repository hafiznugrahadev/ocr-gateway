"""Image preprocessing for scanned documents before OCR.

Pipeline:
  1. grayscale
  2. deskew (correct small rotations)
  3. denoise (fastNlMeans)
  4. CLAHE contrast enhancement
  5. Otsu binarization (only if image is mostly text-like)
  6. upscale if too small (< 1000px wide)

The preprocessor is conservative: if the source image already looks clean,
it returns the original to avoid degrading good scans.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MIN_WIDTH = 1000


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate skew angle from text contours, rotate to correct it."""
    try:
        inv = cv2.bitwise_not(gray)
        thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        # OpenCV returns angle in [-90, 0); normalize.
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5 or abs(angle) > 15:
            # tiny angles are noise; large ones aren't simple skews
            return gray
        h, w = gray.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    except Exception:
        logger.exception("deskew failed; using original")
        return gray


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _maybe_upscale(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if w >= _MIN_WIDTH:
        return gray
    scale = _MIN_WIDTH / w
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)


def _looks_clean(gray: np.ndarray) -> bool:
    """Heuristic: if mean ~ high and stddev moderate, treat as clean printed page."""
    mean = float(gray.mean())
    std = float(gray.std())
    return mean > 200 and 20 < std < 80


def preprocess(rgb_image: np.ndarray) -> np.ndarray:
    """
    Run the preprocessing pipeline and return a 3-channel RGB image
    suitable for PaddleOCR.

    Input/output shape: (H, W, 3) uint8.
    """
    gray = _to_gray(rgb_image)

    if _looks_clean(gray):
        logger.debug("image looks clean; skipping aggressive preprocessing")
        gray = _maybe_upscale(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    gray = _deskew(gray)
    gray = _denoise(gray)
    gray = _clahe(gray)
    gray = _maybe_upscale(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

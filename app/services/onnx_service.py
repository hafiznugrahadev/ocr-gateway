"""ONNX Runtime backend for OCR, exposed via `engine=onnx` on /extract.

Uses RapidOCR (PP-OCRv4 ONNX models) instead of PaddleOCR + Paddle inference.
Typical CPU speedup vs the paddle service: 2-4x. ONNX Runtime sessions are
internally thread-safe and multi-threaded, so we keep a single global engine
instead of a pool -- intra-op threading saturates the CPU just fine on a
single image.

The returned dict matches `ocr_service.ocr_image()` so the /extract router
can swap backends without other changes.
"""

import io
import logging
import threading
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.services.preprocessor import preprocess

logger = logging.getLogger(__name__)

_engine: Any = None
_lock = threading.Lock()


def _build_engine() -> Any:
    """Construct one RapidOCR engine. ONNX Runtime is thread-safe across
    sessions and we use intra-op threads for CPU saturation, so one instance
    is enough -- no pool needed.
    """
    from rapidocr_onnxruntime import RapidOCR

    logger.info(
        "building RapidOCR engine threads=%d unclear=%s",
        settings.OCR_CPU_THREADS,
        settings.OCR_UNCLEAR_THRESHOLD,
    )
    # RapidOCR's per-stage config knobs vary slightly between versions.
    # Stick to the constructor defaults (PP-OCRv4 mobile bundled in the
    # package) and rely on global session options for threading.
    try:
        return RapidOCR(
            text_score=settings.OCR_UNCLEAR_THRESHOLD,
            intra_op_num_threads=settings.OCR_CPU_THREADS,
            inter_op_num_threads=settings.OCR_CPU_THREADS,
        )
    except TypeError:
        # Older API: those kwargs may not be exposed at the top level.
        return RapidOCR()


def get_onnx() -> Any:
    """Lazy-init singleton. Triggered on first /extract request with
    engine=onnx, or at warmup if OCR_DEFAULT_ENGINE=onnx."""
    global _engine
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        _engine = _build_engine()
        _jit_warmup(_engine)
        return _engine


def _jit_warmup(engine: Any) -> None:
    """Run one dummy predict so ONNX Runtime allocates its arenas and we
    don't pay that cost on the first real request."""
    try:
        dummy = np.full((640, 480, 3), 240, dtype=np.uint8)
        dummy[200:230, 80:400] = 30
        dummy[300:330, 80:400] = 30
        engine(dummy)
        logger.info("RapidOCR engine warmup complete")
    except Exception:
        logger.exception("RapidOCR warmup failed; first request may be slow")


def _decode_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        return np.array(img)


def _coerce_bbox(poly: Any) -> list[list[float]]:
    arr = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in arr]


def ocr_image(image_bytes: bytes, lang: str) -> dict[str, Any]:
    """Run OCR via ONNX Runtime. Same return shape as
    `ocr_service.ocr_image()`.
    """
    rgb = _decode_image(image_bytes)
    if settings.OCR_ENABLE_PREPROCESS:
        rgb = preprocess(rgb)

    engine = get_onnx()
    result, _elapse = engine(rgb)

    lines: list[dict[str, Any]] = []
    confidences: list[float] = []
    rendered: list[str] = []

    for item in result or []:
        # RapidOCR returns either [bbox, text, score] tuples or dicts
        # depending on version; normalize both.
        if isinstance(item, dict):
            bbox_raw = item.get("box") or item.get("dt_polys")
            text = item.get("text") or item.get("rec_texts") or ""
            score = float(item.get("score") or item.get("rec_scores") or 0.0)
        else:
            try:
                bbox_raw, text, score = item[0], item[1], float(item[2])
            except (IndexError, TypeError, ValueError):
                continue

        try:
            bbox = _coerce_bbox(bbox_raw) if bbox_raw is not None else []
        except Exception:
            bbox = []

        unclear = score < settings.OCR_UNCLEAR_THRESHOLD
        lines.append(
            {
                "text": str(text),
                "confidence": score,
                "bbox": bbox,
                "unclear": unclear,
            }
        )
        confidences.append(score)
        rendered.append("[UNCLEAR]" if unclear else str(text))

    text = "\n".join(rendered)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return {"text": text, "lines": lines, "confidence": avg_conf}

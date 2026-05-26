"""PaddleOCR 3.x wrapper with a per-language engine pool.

PaddleOCR instances are not safe for concurrent predict() calls (the
underlying Paddle inference predictor holds mutable graph state). To OCR
multiple PDF pages in parallel we keep a small pool of independent engines
per language and acquire one per call.
"""

import io
import logging
import queue as _queue
import threading
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

_engine_pools: dict[str, _queue.Queue] = {}
_init_lock = threading.Lock()


def _build_engine(lang: str) -> Any:
    """Construct one PaddleOCR engine.

    Heavy: 200-500MB RAM, 5-15s cold load. Called N times during pool init.
    """
    from paddleocr import PaddleOCR

    device = "gpu:0" if settings.OCR_USE_GPU else "cpu"
    per_engine_threads = max(
        1, settings.OCR_CPU_THREADS // settings.OCR_PARALLEL_WORKERS
    )
    logger.info(
        "building PaddleOCR engine lang=%s device=%s det_model=%s mkldnn=%s threads=%d",
        lang,
        device,
        settings.OCR_TEXT_DETECTION_MODEL,
        settings.OCR_ENABLE_MKLDNN,
        per_engine_threads,
    )
    kwargs: dict[str, Any] = {
        "lang": lang,
        "device": device,
        "use_doc_orientation_classify": settings.OCR_USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": settings.OCR_USE_DOC_UNWARPING,
        "use_textline_orientation": settings.OCR_USE_ANGLE_CLS,
        "text_detection_model_name": settings.OCR_TEXT_DETECTION_MODEL,
        # Authoritative MKLDNN/OneDNN switch for PaddleOCR 3.x PIR path.
        # FLAGS_use_mkldnn env var alone is NOT honored by the new PaddleX
        # inference runner and triggers ConvertPirAttribute2RuntimeAttribute
        # crashes on some images.
        "enable_mkldnn": settings.OCR_ENABLE_MKLDNN,
        "cpu_threads": per_engine_threads,
    }
    try:
        return PaddleOCR(**kwargs)
    except TypeError as exc:
        # PaddleOCR 3.x has renamed some params across minor versions.
        # Drop optional kwargs progressively and retry.
        logger.warning(
            "PaddleOCR init failed with full kwargs (%s); retrying minimal", exc
        )
        try:
            return PaddleOCR(
                lang=lang,
                device=device,
                use_textline_orientation=settings.OCR_USE_ANGLE_CLS,
            )
        except TypeError:
            return PaddleOCR(
                lang=lang,
                use_textline_orientation=settings.OCR_USE_ANGLE_CLS,
            )


def _ensure_pool(lang: str) -> _queue.Queue:
    """Lazily provision OCR_PARALLEL_WORKERS engines for `lang`."""
    pool = _engine_pools.get(lang)
    if pool is not None:
        return pool
    with _init_lock:
        pool = _engine_pools.get(lang)
        if pool is not None:
            return pool
        size = settings.OCR_PARALLEL_WORKERS
        logger.info("provisioning OCR engine pool lang=%s size=%d", lang, size)
        new_pool: _queue.Queue = _queue.Queue()
        for _ in range(size):
            new_pool.put(_build_engine(lang))
        _engine_pools[lang] = new_pool
        return new_pool


def get_ocr(lang: str) -> None:
    """Ensure the engine pool for `lang` is provisioned. Used at warmup."""
    _ensure_pool(lang)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        return np.array(img)


def _coerce_bbox(poly: Any) -> list[list[float]]:
    arr = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in arr]


def ocr_image(image_bytes: bytes, lang: str) -> dict[str, Any]:
    """Run OCR on a single image. Returns {text, lines, confidence}.

    Acquires an engine from the per-language pool (blocks if all are busy),
    runs predict, returns it to the pool. Lines with confidence
    < OCR_UNCLEAR_THRESHOLD are marked as [UNCLEAR] in the joined `text`,
    while the original recognized string is preserved in `lines[i].text`.
    """
    pool = _ensure_pool(lang)
    engine = pool.get()
    try:
        rgb = _decode_image(image_bytes)
        raw = engine.predict(rgb)
    finally:
        pool.put(engine)

    lines: list[dict[str, Any]] = []
    confidences: list[float] = []
    rendered: list[str] = []

    for res in raw or []:
        data = res.json if hasattr(res, "json") else res
        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        texts = data.get("rec_texts", []) or []
        scores = data.get("rec_scores", []) or []
        polys = data.get("rec_polys") or data.get("dt_polys") or []

        for idx, text in enumerate(texts):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            poly = polys[idx] if idx < len(polys) else []
            try:
                bbox = _coerce_bbox(poly) if len(poly) else []
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

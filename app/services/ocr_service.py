"""PaddleOCR 3.x wrapper. Singleton per language."""

import io
import logging
import threading
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

_engines: dict[str, Any] = {}
_lock = threading.Lock()


def get_ocr(lang: str) -> Any:
    if lang in _engines:
        return _engines[lang]
    with _lock:
        if lang in _engines:
            return _engines[lang]
        from paddleocr import PaddleOCR

        device = "gpu:0" if settings.OCR_USE_GPU else "cpu"
        logger.info(
            "loading PaddleOCR lang=%s device=%s det_model=%s mkldnn=%s threads=%d",
            lang,
            device,
            settings.OCR_TEXT_DETECTION_MODEL,
            settings.OCR_ENABLE_MKLDNN,
            settings.OCR_CPU_THREADS,
        )
        kwargs: dict[str, Any] = {
            "lang": lang,
            "device": device,
            "use_doc_orientation_classify": settings.OCR_USE_DOC_ORIENTATION_CLASSIFY,
            "use_doc_unwarping": settings.OCR_USE_DOC_UNWARPING,
            "use_textline_orientation": settings.OCR_USE_ANGLE_CLS,
            "text_detection_model_name": settings.OCR_TEXT_DETECTION_MODEL,
            # Authoritative MKLDNN/OneDNN switch for PaddleOCR 3.x PIR path.
            # FLAGS_use_mkldnn env var alone is NOT honored by the new
            # PaddleX inference runner and triggers
            # ConvertPirAttribute2RuntimeAttribute crashes on some images.
            "enable_mkldnn": settings.OCR_ENABLE_MKLDNN,
            "cpu_threads": settings.OCR_CPU_THREADS,
        }
        try:
            engine = PaddleOCR(**kwargs)
        except TypeError as exc:
            # PaddleOCR 3.x has renamed some params across minor versions.
            # Drop optional kwargs progressively and retry.
            logger.warning("PaddleOCR init failed with full kwargs (%s); retrying minimal", exc)
            try:
                engine = PaddleOCR(
                    lang=lang,
                    device=device,
                    use_textline_orientation=settings.OCR_USE_ANGLE_CLS,
                )
            except TypeError:
                engine = PaddleOCR(
                    lang=lang,
                    use_textline_orientation=settings.OCR_USE_ANGLE_CLS,
                )
        _engines[lang] = engine
        return engine


def _decode_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        return np.array(img)


def _coerce_bbox(poly: Any) -> list[list[float]]:
    arr = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in arr]


def ocr_image(image_bytes: bytes, lang: str) -> dict[str, Any]:
    """Run OCR on a single image. Returns {text, lines, confidence}.

    Lines with confidence < OCR_UNCLEAR_THRESHOLD are marked as [UNCLEAR]
    in the joined `text`, while the original recognized string is preserved
    in `lines[i].text` for inspection.
    """
    engine = get_ocr(lang)
    rgb = _decode_image(image_bytes)

    raw = engine.predict(rgb)

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

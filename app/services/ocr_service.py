"""PaddleOCR 3.x wrapper with a single global engine pool.

PaddleOCR instances are not safe for concurrent predict() calls (the
underlying Paddle inference predictor holds mutable graph state). To OCR
multiple PDF pages in parallel we keep a small pool of independent engines
and acquire one per call.

The pool is keyed globally (not per-language) because we always pass
explicit detection/recognition model names; PaddleOCR ignores the `lang`
kwarg in that mode, so every incoming `language=` query would otherwise
spin up a redundant pool. The `lang` argument is still threaded through
for logging and as a placeholder for future per-language model selection.
"""

import io
import logging
import queue as _queue
import threading
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.services.preprocessor import preprocess

logger = logging.getLogger(__name__)

_GLOBAL_POOL_KEY = "default"
_engine_pools: dict[str, _queue.Queue] = {}
_init_lock = threading.Lock()


def _build_engine(lang: str) -> Any:
    """Construct one PaddleOCR engine and trigger PIR JIT compile.

    Heavy: 200-500MB RAM, 5-15s cold load + 5-30s JIT. Called N times during
    pool init. Doing the JIT warmup here (single-threaded) keeps it off the
    first live request, where 4 parallel JIT compiles would compete for CPU.
    """
    from paddleocr import PaddleOCR

    device = "gpu:0" if settings.OCR_USE_GPU else "cpu"
    per_engine_threads = max(
        1, settings.OCR_CPU_THREADS // settings.OCR_PARALLEL_WORKERS
    )
    logger.info(
        "building PaddleOCR engine lang=%s device=%s det=%s rec=%s mkldnn=%s threads=%d",
        lang,
        device,
        settings.OCR_TEXT_DETECTION_MODEL,
        settings.OCR_TEXT_RECOGNITION_MODEL,
        settings.OCR_ENABLE_MKLDNN,
        per_engine_threads,
    )
    # Note: PaddleOCR ignores `lang` when explicit model names are given, so we
    # rely on the multilingual mobile_rec model handling Indonesian / Latin
    # scripts. Override OCR_TEXT_RECOGNITION_MODEL if you need a language-
    # specific variant.
    kwargs: dict[str, Any] = {
        "lang": lang,
        "device": device,
        "use_doc_orientation_classify": settings.OCR_USE_DOC_ORIENTATION_CLASSIFY,
        "use_doc_unwarping": settings.OCR_USE_DOC_UNWARPING,
        "use_textline_orientation": settings.OCR_USE_ANGLE_CLS,
        "text_detection_model_name": settings.OCR_TEXT_DETECTION_MODEL,
        "text_recognition_model_name": settings.OCR_TEXT_RECOGNITION_MODEL,
        # Authoritative MKLDNN/OneDNN switch for PaddleOCR 3.x PIR path.
        # FLAGS_use_mkldnn env var alone is NOT honored by the new PaddleX
        # inference runner and triggers ConvertPirAttribute2RuntimeAttribute
        # crashes on some images.
        "enable_mkldnn": settings.OCR_ENABLE_MKLDNN,
        "cpu_threads": per_engine_threads,
    }
    try:
        engine = PaddleOCR(**kwargs)
    except TypeError as exc:
        # PaddleOCR 3.x has renamed some params across minor versions.
        # Drop optional kwargs progressively and retry.
        logger.warning(
            "PaddleOCR init failed with full kwargs (%s); retrying minimal", exc
        )
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

    _jit_warmup(engine)
    return engine


def _jit_warmup(engine: Any) -> None:
    """Force PaddleOCR's PIR executor to JIT-compile by running a dummy
    predict. Without this, the first live request pays a 5-30s compile cost
    that multiplies across the pool because N engines compile in parallel.
    """
    try:
        # Synthetic page-sized image with some "text-like" black bars so the
        # detection model finds at least one box and triggers recognition JIT.
        dummy = np.full((640, 480, 3), 240, dtype=np.uint8)
        dummy[200:230, 80:400] = 30
        dummy[300:330, 80:400] = 30
        engine.predict(dummy)
        logger.info("engine JIT warmup complete")
    except Exception:
        logger.exception("engine JIT warmup failed; first request may be slow")


def _ensure_pool(_lang: str = "") -> _queue.Queue:
    """Lazily provision OCR_PARALLEL_WORKERS engines.

    The pool is global; `_lang` is accepted only for API compatibility and
    is otherwise ignored. See module docstring.
    """
    pool = _engine_pools.get(_GLOBAL_POOL_KEY)
    if pool is not None:
        return pool
    with _init_lock:
        pool = _engine_pools.get(_GLOBAL_POOL_KEY)
        if pool is not None:
            return pool
        size = settings.OCR_PARALLEL_WORKERS
        logger.info(
            "provisioning OCR engine pool size=%d (warmup includes JIT)", size
        )
        new_pool: _queue.Queue = _queue.Queue()
        for _ in range(size):
            new_pool.put(_build_engine(settings.OCR_LANGUAGE))
        _engine_pools[_GLOBAL_POOL_KEY] = new_pool
        return new_pool


def get_ocr(lang: str) -> None:
    """Ensure the engine pool is provisioned. Used at warmup."""
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

    Acquires an engine from the global pool (blocks if all are busy),
    runs predict, returns it to the pool. Lines with confidence
    < OCR_UNCLEAR_THRESHOLD are marked as [UNCLEAR] in the joined `text`,
    while the original recognized string is preserved in `lines[i].text`.
    """
    rgb = _decode_image(image_bytes)
    if settings.OCR_ENABLE_PREPROCESS:
        rgb = preprocess(rgb)

    pool = _ensure_pool(lang)
    engine = pool.get()
    try:
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

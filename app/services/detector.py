"""Detect whether a PDF has a text layer (digital) vs scan/image."""

import logging

import fitz

from app.config import settings

logger = logging.getLogger(__name__)


def has_text_layer(pdf_bytes: bytes, sample_pages: int = 3) -> tuple[bool, int]:
    """
    Decide whether `pdf_bytes` is a text-based PDF (has extractable text layer)
    or a scan/image PDF that needs OCR.

    Strategy: sample up to `sample_pages` pages, sum extracted text length.
    If average chars/page >= OCR_MIN_TEXT_LENGTH, consider it text-based.

    Returns (is_text_based, total_pages).
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = doc.page_count
        if total_pages == 0:
            return False, 0

        n = min(sample_pages, total_pages)
        # Sample evenly: first, middle, last
        if n == 1:
            indices = [0]
        elif n == 2:
            indices = [0, total_pages - 1]
        else:
            indices = [0, total_pages // 2, total_pages - 1][:n]

        chars_total = 0
        for i in indices:
            page = doc.load_page(i)
            text = (page.get_text("text") or "").strip()
            chars_total += len(text)
            logger.debug("text-layer probe page=%d chars=%d", i + 1, len(text))

        avg = chars_total / max(1, len(indices))
        is_text = avg >= settings.OCR_MIN_TEXT_LENGTH
        logger.info(
            "pdf detected total_pages=%d sampled=%d avg_chars=%.0f is_text_based=%s",
            total_pages,
            len(indices),
            avg,
            is_text,
        )
        return is_text, total_pages

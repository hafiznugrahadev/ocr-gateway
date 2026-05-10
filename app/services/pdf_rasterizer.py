"""Rasterize selected PDF pages to PNG bytes (for OCR path)."""

import logging

import fitz

from app.utils.errors import OcrGatewayError

logger = logging.getLogger(__name__)


def rasterize_pages(pdf_bytes: bytes, page_numbers: list[int], dpi: int = 300) -> list[tuple[int, bytes]]:
    """Return list of (page_number, png_bytes) for each requested 1-based page."""
    out: list[tuple[int, bytes]] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for n in page_numbers:
                idx = n - 1
                if idx < 0 or idx >= doc.page_count:
                    continue
                page = doc.load_page(idx)
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                out.append((n, pix.tobytes("png")))
    except OcrGatewayError:
        raise
    except Exception as exc:
        raise OcrGatewayError(
            error_code="INVALID_PDF",
            detail=f"Failed to read PDF: {exc}",
            status_code=422,
        ) from exc
    return out

"""Extract text from text-based PDFs without OCR (uses PyMuPDF text layer)."""

import logging

import fitz

logger = logging.getLogger(__name__)


def extract_text_layer(pdf_bytes: bytes, page_numbers: list[int]) -> list[dict]:
    """
    Extract text per page from a digital (text-based) PDF.

    `page_numbers` is 1-based.

    Returns a list of dicts: [{"page": n, "text": "...", "confidence": 1.0,
    "lines": [...]}].

    For text-layer extraction we report confidence=1.0 (not from OCR).
    Each "line" is a (text, bbox) pair derived from PyMuPDF blocks/lines.
    """
    out: list[dict] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for n in page_numbers:
            idx = n - 1
            if idx < 0 or idx >= doc.page_count:
                continue
            page = doc.load_page(idx)
            text = (page.get_text("text") or "").rstrip()

            lines: list[dict] = []
            try:
                blocks = page.get_text("dict").get("blocks", [])
                for b in blocks:
                    for ln in b.get("lines", []) or []:
                        ln_text = "".join(span.get("text", "") for span in ln.get("spans", [])).strip()
                        if not ln_text:
                            continue
                        bbox = ln.get("bbox", [0, 0, 0, 0])
                        x0, y0, x1, y1 = bbox
                        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                        lines.append(
                            {
                                "text": ln_text,
                                "confidence": 1.0,
                                "bbox": [[float(x), float(y)] for x, y in poly],
                            }
                        )
            except Exception:
                logger.exception("failed to parse text dict on page=%d", n)

            out.append(
                {
                    "page": n,
                    "text": text,
                    "confidence": 1.0,
                    "lines": lines,
                }
            )
    return out

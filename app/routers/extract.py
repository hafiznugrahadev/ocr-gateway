import asyncio
import json
import logging
import re
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.config import settings
from app.dependencies import require_bearer
from app.models.schemas import (
    ExtractFormParams,
    ExtractMetadata,
    ExtractResponse,
    ExtractResult,
    OutputFormat,
    PageResult,
    UrlExtractRequest,
)
from app.services.detector import has_text_layer
from app.services.ocr_service import get_ocr, ocr_image
from app.services.pdf_extractor import extract_text_layer
from app.services.pdf_rasterizer import rasterize_pages
from app.services.url_fetcher import fetch_url
from app.utils.errors import OcrGatewayError
from app.utils.pages import parse_pages

logger = logging.getLogger(__name__)
router = APIRouter(tags=["OCR"])

_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/tiff",
    "image/bmp",
}
_PDF_MIME = "application/pdf"
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _detect_kind(content_type: str | None, head: bytes) -> str:
    if head.startswith(b"%PDF"):
        return "pdf"
    if content_type:
        ct = content_type.lower().split(";", 1)[0].strip()
        if ct == _PDF_MIME:
            return "pdf"
        if ct in _IMAGE_MIMES:
            return "image"
    return ""


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _build_full_text(pages: list[PageResult], output_format: OutputFormat) -> str:
    if output_format == "text":
        return "\n\n---\n\n".join(p.text for p in pages)
    if output_format == "markdown":
        chunks: list[str] = []
        for p in pages:
            chunks.append(f"## Halaman {p.page}\n\n{p.text}")
        return "\n\n---\n\n".join(chunks)
    return "\n\n".join(p.text for p in pages)


async def _read_inputs(
    request: Request,
    file: UploadFile | None,
    url: str | None,
    language: str | None,
    pages: str | None,
    output_format: str | None,
) -> tuple[bytes, str | None, str, str, str, str, str]:
    """
    Resolve raw input bytes + meta from either multipart upload, multipart URL,
    or JSON body. Returns (raw, content_type, filename, source, lang, pages_spec,
    output_format).

    All structured fields are validated via Pydantic models. Invalid values
    raise pydantic.ValidationError, which the global handler turns into a
    structured 422 response.
    """
    content_type_header = (request.headers.get("content-type") or "").lower()

    if content_type_header.startswith("application/json"):
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise OcrGatewayError("INVALID_JSON", f"Invalid JSON: {exc}", 400) from exc
        if not isinstance(body, dict):
            raise OcrGatewayError("INVALID_JSON", "Body must be a JSON object", 400)

        params = UrlExtractRequest.model_validate(body)
        chosen_lang = params.language or settings.OCR_LANGUAGE
        raw, content_type, filename = await fetch_url(str(params.url))
        return (
            raw,
            content_type,
            filename,
            "url",
            chosen_lang,
            params.pages,
            params.output_format,
        )

    form_params = ExtractFormParams.model_validate(
        {
            "url": url,
            "language": language,
            "pages": pages,
            "output_format": output_format,
        }
    )

    has_file = bool(file is not None and (file.filename or "").strip())
    has_url = form_params.url is not None
    if has_file and has_url:
        raise OcrGatewayError(
            "BOTH_INPUT", "Provide exactly one of: 'file' or 'url'", 400
        )
    if not has_file and not has_url:
        raise OcrGatewayError(
            "MISSING_INPUT", "Provide one of: 'file' (multipart) or 'url'", 400
        )

    chosen_lang = form_params.language or settings.OCR_LANGUAGE

    if has_url:
        raw, content_type, filename = await fetch_url(str(form_params.url))
        return (
            raw,
            content_type,
            filename,
            "url",
            chosen_lang,
            form_params.pages,
            form_params.output_format,
        )

    raw = await file.read()
    return (
        raw,
        file.content_type,
        file.filename or "",
        "upload",
        chosen_lang,
        form_params.pages,
        form_params.output_format,
    )


@router.post("/extract", response_model=ExtractResponse)
async def extract_endpoint(
    request: Request,
    _token: str = Depends(require_bearer),
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    pages: Annotated[str | None, Form()] = None,
    output_format: Annotated[str | None, Form()] = None,
) -> ExtractResponse:
    started = time.perf_counter()

    raw, content_type, filename, source, chosen_lang, pages_spec, chosen_format = await _read_inputs(
        request, file, url, language, pages, output_format
    )

    if not raw:
        raise OcrGatewayError("MISSING_INPUT", "Empty file", 400)
    if len(raw) > settings.max_file_bytes:
        raise OcrGatewayError(
            "FILE_TOO_LARGE",
            f"File size {len(raw) // (1024 * 1024)}MB exceeds maximum {settings.OCR_MAX_FILE_SIZE_MB}MB",
            413,
        )

    kind = _detect_kind(content_type, raw[:8])
    if not kind:
        raise OcrGatewayError(
            "UNSUPPORTED_FORMAT",
            f"Unsupported content type: {content_type or 'unknown'}",
            415,
        )

    file_size_bytes = len(raw)
    logger.info(
        "extract received source=%s filename=%s size=%dKB type=%s lang=%s format=%s pages=%s",
        source,
        filename or "?",
        file_size_bytes // 1024,
        content_type or "?",
        chosen_lang,
        chosen_format,
        pages_spec,
    )

    page_results: list[PageResult] = []
    method: str

    if kind == "pdf":
        try:
            is_text_based, total_pages = await asyncio.to_thread(has_text_layer, raw)
        except Exception as exc:
            raise OcrGatewayError("INVALID_PDF", f"Failed to read PDF: {exc}", 422) from exc

        if total_pages == 0:
            raise OcrGatewayError("INVALID_PDF", "PDF has 0 pages", 422)
        if total_pages > settings.OCR_MAX_PAGES:
            raise OcrGatewayError(
                "TOO_MANY_PAGES",
                f"PDF has {total_pages} pages, exceeds limit of {settings.OCR_MAX_PAGES}",
                400,
            )

        selected = parse_pages(pages_spec, total_pages)
        if not selected:
            raise OcrGatewayError("INVALID_PAGES", "No pages selected", 400)

        if is_text_based:
            method = "text-layer"
            logger.info("pdf using text-layer extraction pages=%d", len(selected))
            extracted = await asyncio.to_thread(extract_text_layer, raw, selected)
            for item in extracted:
                page_results.append(
                    PageResult(
                        page=item["page"],
                        text=item["text"],
                        confidence=float(item.get("confidence", 1.0)),
                        word_count=_word_count(item["text"]),
                        has_table=False,
                        has_unclear=False,
                    )
                )
        else:
            method = "ocr"
            logger.info("pdf using OCR pages=%d", len(selected))
            rasters = await asyncio.to_thread(
                rasterize_pages, raw, selected, settings.OCR_PDF_DPI
            )
            for n, png_bytes in rasters:
                logger.info("ocr pdf page=%d processing", n)
                result = await asyncio.to_thread(ocr_image, png_bytes, chosen_lang)
                has_unclear = any(line.get("unclear") for line in result.get("lines", []))
                page_results.append(
                    PageResult(
                        page=n,
                        text=result["text"],
                        confidence=float(result.get("confidence", 0.0)),
                        word_count=_word_count(result["text"]),
                        has_table=False,
                        has_unclear=has_unclear,
                    )
                )
    else:
        method = "ocr"
        try:
            result = await asyncio.to_thread(ocr_image, raw, chosen_lang)
        except Exception as exc:
            raise OcrGatewayError("OCR_FAILED", f"Image OCR failed: {exc}", 500) from exc
        has_unclear = any(line.get("unclear") for line in result.get("lines", []))
        page_results.append(
            PageResult(
                page=1,
                text=result["text"],
                confidence=float(result.get("confidence", 0.0)),
                word_count=_word_count(result["text"]),
                has_table=False,
                has_unclear=has_unclear,
            )
        )
        total_pages = 1

    full_text = _build_full_text(page_results, chosen_format)
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "extract completed filename=%s method=%s pages_processed=%d chars=%d duration_ms=%d",
        filename or "?",
        method,
        len(page_results),
        len(full_text),
        duration_ms,
    )

    # Eagerly load default OCR engine on first call so subsequent requests are warm.
    # (Health endpoint also flips ready=True from the lifespan task.)
    _ = get_ocr  # noqa: F841 (referenced for warmup; engine already cached)

    return ExtractResponse(
        success=True,
        engine="paddleocr",
        method=method,
        pages_processed=len(page_results),
        total_pages=total_pages,
        output_format=chosen_format,
        result=ExtractResult(
            pages=page_results,
            full_text=full_text,
            metadata=ExtractMetadata(
                file_name=filename or "",
                file_size_bytes=file_size_bytes,
                processing_time_ms=duration_ms,
                source=source,
            ),
        ),
    )

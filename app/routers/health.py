from importlib import metadata as importlib_metadata

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.dependencies import require_bearer
from app.models.schemas import FormatsResponse, HealthResponse

router = APIRouter(tags=["System"])

_state: dict[str, bool] = {"ready": False}


def mark_ready() -> None:
    _state["ready"] = True


def is_ready() -> bool:
    return _state["ready"]


def _engine_version() -> str:
    try:
        return importlib_metadata.version("paddleocr")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


@router.get("/health", response_model=HealthResponse)
async def health() -> JSONResponse:
    body = {
        "status": "healthy" if _state["ready"] else "warming",
        "version": "1.0.0",
        "engine": "paddleocr",
        "engine_version": _engine_version(),
        "gpu_available": settings.OCR_USE_GPU,
        "mkldnn_enabled": settings.OCR_ENABLE_MKLDNN,
        "supported_formats": ["pdf", "jpg", "jpeg", "png", "tiff", "bmp", "webp"],
        "max_file_size_mb": settings.OCR_MAX_FILE_SIZE_MB,
        "max_pages": settings.OCR_MAX_PAGES,
    }
    return JSONResponse(
        body,
        status_code=status.HTTP_200_OK if _state["ready"] else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@router.get("/formats", response_model=FormatsResponse)
async def formats(_token: str = Depends(require_bearer)) -> FormatsResponse:
    return FormatsResponse(
        input_formats={
            "documents": ["pdf"],
            "images": ["jpg", "jpeg", "png", "tiff", "bmp", "webp"],
        },
        output_formats={
            "text": "Plain text, pages separated by ---",
            "markdown": "Markdown with preserved structure",
            "json": "Structured JSON with per-page metadata",
        },
        input_methods={
            "upload": "multipart/form-data file upload",
            "url": "HTTP/HTTPS URL to file (presigned URLs supported)",
        },
    )

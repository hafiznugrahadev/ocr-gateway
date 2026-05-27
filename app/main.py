import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import settings
from app.routers import extract as extract_router
from app.routers import health as health_router
from app.services.ocr_service import get_ocr
from app.utils.errors import OcrGatewayError, error_response, gateway_error_handler

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ocr-gateway")


async def _warm_default_model() -> None:
    try:
        logger.info("warming default OCR model lang=%s", settings.OCR_LANGUAGE)
        await asyncio.to_thread(get_ocr, settings.OCR_LANGUAGE)
        health_router.mark_ready()
        logger.info("OCR gateway ready")
    except Exception:
        logger.exception("OCR warmup failed; first /extract request will retry lazily")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    asyncio.create_task(_warm_default_model())
    yield


app = FastAPI(
    title="OCR Gateway",
    description=(
        "Self-hosted OCR microservice powered by PaddleOCR.\n\n"
        "Extract text from PDF documents and images via file upload or URL.\n\n"
        "**Authentication:** All endpoints (except `/health` and docs) require Bearer Token.\n\n"
        "Header: `Authorization: Bearer <OCR_API_KEY>`\n\n"
        "**Supported input:** PDF, JPG, PNG, TIFF, BMP, WebP. Either multipart upload "
        "(`file`) or HTTP/HTTPS URL (`url` form field, or JSON body)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "OCR", "description": "Text extraction endpoints"},
        {"name": "System", "description": "Health check and system info"},
    ],
    lifespan=lifespan,
)


@app.exception_handler(OcrGatewayError)
async def _gateway_exc(request: Request, exc: OcrGatewayError):
    return await gateway_error_handler(request, exc)


@app.exception_handler(HTTPException)
async def _http_exc(_request: Request, exc: HTTPException) -> JSONResponse:
    code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        408: "REQUEST_TIMEOUT",
        413: "FILE_TOO_LARGE",
        415: "UNSUPPORTED_FORMAT",
        422: "UNPROCESSABLE_ENTITY",
        500: "INTERNAL_ERROR",
        504: "GATEWAY_TIMEOUT",
    }
    return error_response(
        code_map.get(exc.status_code, "ERROR"),
        str(exc.detail),
        exc.status_code,
    )


def _format_validation_errors(errors: list[dict]) -> str:
    parts: list[str] = []
    for err in errors:
        loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        msg = err.get("msg", "")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "Invalid request"


@app.exception_handler(RequestValidationError)
async def _validation_exc(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response("VALIDATION_ERROR", _format_validation_errors(exc.errors()), 422)


@app.exception_handler(ValidationError)
async def _pydantic_validation_exc(_request: Request, exc: ValidationError) -> JSONResponse:
    return error_response("VALIDATION_ERROR", _format_validation_errors(exc.errors()), 422)


@app.exception_handler(Exception)
async def _unhandled_exc(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception during request: %s", exc)
    return error_response("INTERNAL_ERROR", "Internal server error", 500)


app.include_router(health_router.router)
app.include_router(extract_router.router)

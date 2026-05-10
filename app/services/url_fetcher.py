import logging
from urllib.parse import unquote, urlparse

import httpx

from app.config import settings
from app.utils.errors import OcrGatewayError

logger = logging.getLogger(__name__)


async def fetch_url(url: str) -> tuple[bytes, str | None, str]:
    """Fetch bytes from URL with size cap. Returns (bytes, content_type, filename)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise OcrGatewayError(
            error_code="INVALID_URL",
            detail=f"Unsupported URL scheme: {parsed.scheme or '(empty)'}. Use http or https.",
            status_code=400,
        )
    if not parsed.netloc:
        raise OcrGatewayError(
            error_code="INVALID_URL",
            detail="URL is missing host",
            status_code=400,
        )

    filename = unquote(parsed.path.rsplit("/", 1)[-1]) or "remote_file"
    cap = settings.max_file_bytes
    timeout = httpx.Timeout(
        connect=10.0,
        read=float(settings.OCR_URL_DOWNLOAD_TIMEOUT),
        write=10.0,
        pool=10.0,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise OcrGatewayError(
                        error_code="URL_UNREACHABLE",
                        detail=f"Fetch failed: HTTP {resp.status_code}",
                        status_code=422,
                    )
                content_type = resp.headers.get("content-type")
                buf = bytearray()
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    buf.extend(chunk)
                    if len(buf) > cap:
                        raise OcrGatewayError(
                            error_code="FILE_TOO_LARGE",
                            detail=f"Remote file exceeds {settings.OCR_MAX_FILE_SIZE_MB} MB limit",
                            status_code=413,
                        )
                return bytes(buf), content_type, filename
    except OcrGatewayError:
        raise
    except httpx.TimeoutException as exc:
        raise OcrGatewayError(
            error_code="URL_TIMEOUT",
            detail=f"Timed out after {settings.OCR_URL_DOWNLOAD_TIMEOUT}s fetching URL",
            status_code=408,
        ) from exc
    except httpx.HTTPError as exc:
        raise OcrGatewayError(
            error_code="URL_UNREACHABLE",
            detail=f"Failed to fetch URL: {exc}",
            status_code=422,
        ) from exc

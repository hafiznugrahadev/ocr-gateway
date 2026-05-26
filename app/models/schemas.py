from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


OutputFormat = Literal["text", "markdown", "json"]
ExtractMethod = Literal["ocr", "text-layer"]
SourceType = Literal["upload", "url"]
EngineName = Literal["paddle", "onnx"]

_PAGES_PATTERN = r"^(all|\d+(-\d+)?(,\d+(-\d+)?)*)$"
_LANGUAGE_PATTERN = r"^[a-z][a-z0-9_]{0,15}$"

_PAGES_DESCRIPTION = (
    "Page selector. Use 'all', a single page ('3'), a range ('1-5'), "
    "or a comma list ('1,3,5-7')."
)
_LANGUAGE_DESCRIPTION = (
    "OCR language code (lowercase, e.g. 'en', 'id', 'ch', 'japan', 'korean')."
)


def _blank_to_none(v: object) -> object | None:
    return None if isinstance(v, str) and not v.strip() else v


def _normalize_output_format(v: object) -> object:
    if v is None:
        return "json"
    if isinstance(v, str):
        stripped = v.strip().lower()
        return stripped or "json"
    return v


def _normalize_engine(v: object) -> object | None:
    if v is None:
        return None
    if isinstance(v, str):
        stripped = v.strip().lower()
        return stripped or None
    return v


def _normalize_pages(v: object) -> object:
    if v is None:
        return "all"
    if isinstance(v, str):
        stripped = v.strip()
        return stripped or "all"
    return v


_ENGINE_DESCRIPTION = (
    "OCR backend: 'paddle' (PaddleOCR PP-OCRv5 mobile) or 'onnx' "
    "(RapidOCR PP-OCRv4 via ONNX Runtime). Leave unset to use the server "
    "default (OCR_DEFAULT_ENGINE)."
)


class ExtractFormParams(BaseModel):
    """Validated text fields for multipart /extract.

    The binary `file` upload is validated by FastAPI/Starlette separately;
    this model covers the structured text inputs only.
    """

    url: HttpUrl | None = None
    language: str | None = Field(
        default=None, pattern=_LANGUAGE_PATTERN, description=_LANGUAGE_DESCRIPTION
    )
    pages: str = Field(
        default="all", pattern=_PAGES_PATTERN, description=_PAGES_DESCRIPTION
    )
    output_format: OutputFormat = "json"
    engine: EngineName | None = Field(default=None, description=_ENGINE_DESCRIPTION)

    @field_validator("url", "language", mode="before")
    @classmethod
    def _strip_blank_optional(cls, v: object) -> object | None:
        return _blank_to_none(v)

    @field_validator("pages", mode="before")
    @classmethod
    def _default_pages(cls, v: object) -> object:
        return _normalize_pages(v)

    @field_validator("output_format", mode="before")
    @classmethod
    def _default_output_format(cls, v: object) -> object:
        return _normalize_output_format(v)

    @field_validator("engine", mode="before")
    @classmethod
    def _strip_blank_engine(cls, v: object) -> object | None:
        return _normalize_engine(v)


class UrlExtractRequest(BaseModel):
    """JSON body for /extract when sending a URL."""

    url: HttpUrl
    language: str | None = Field(
        default=None, pattern=_LANGUAGE_PATTERN, description=_LANGUAGE_DESCRIPTION
    )
    pages: str = Field(
        default="all", pattern=_PAGES_PATTERN, description=_PAGES_DESCRIPTION
    )
    output_format: OutputFormat = "json"
    engine: EngineName | None = Field(default=None, description=_ENGINE_DESCRIPTION)

    @field_validator("language", mode="before")
    @classmethod
    def _strip_blank_language(cls, v: object) -> object | None:
        return _blank_to_none(v)

    @field_validator("pages", mode="before")
    @classmethod
    def _default_pages(cls, v: object) -> object:
        return _normalize_pages(v)

    @field_validator("output_format", mode="before")
    @classmethod
    def _default_output_format(cls, v: object) -> object:
        return _normalize_output_format(v)

    @field_validator("engine", mode="before")
    @classmethod
    def _strip_blank_engine(cls, v: object) -> object | None:
        return _normalize_engine(v)


class PageResult(BaseModel):
    page: int = Field(ge=1)
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    word_count: int = Field(ge=0)
    has_table: bool = False
    has_unclear: bool = False


class ExtractMetadata(BaseModel):
    file_name: str
    file_size_bytes: int
    processing_time_ms: int
    source: SourceType


class ExtractResult(BaseModel):
    pages: list[PageResult]
    full_text: str
    metadata: ExtractMetadata


class ExtractResponse(BaseModel):
    success: bool = True
    engine: str = "paddleocr"
    method: ExtractMethod
    pages_processed: int
    total_pages: int
    output_format: OutputFormat
    result: ExtractResult


class HealthResponse(BaseModel):
    status: str
    version: str
    engine: str
    engine_version: str
    gpu_available: bool
    mkldnn_enabled: bool
    supported_formats: list[str]
    max_file_size_mb: int
    max_pages: int


class FormatsResponse(BaseModel):
    input_formats: dict[str, list[str]]
    output_formats: dict[str, str]
    input_methods: dict[str, str]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str
    status_code: int

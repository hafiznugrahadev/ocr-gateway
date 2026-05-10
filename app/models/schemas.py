from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


OutputFormat = Literal["text", "markdown", "json"]
ExtractMethod = Literal["ocr", "text-layer"]
SourceType = Literal["upload", "url"]


class UrlExtractRequest(BaseModel):
    """JSON body for /extract when sending a URL."""

    url: HttpUrl
    output_format: OutputFormat = "json"
    language: str | None = None
    pages: str = "all"


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

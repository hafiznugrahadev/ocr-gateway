import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    OCR_API_KEY: str = Field(..., min_length=1)

    OCR_LANGUAGE: str = "en"
    OCR_USE_GPU: bool = False
    # MKLDNN currently disabled by default: PaddleOCR 3.x + Paddle 3.x PIR path
    # has unimplemented OneDNN op coverage that crashes on some images. Re-enable
    # after upstream fixes the ConvertPirAttribute2RuntimeAttribute gap.
    OCR_ENABLE_MKLDNN: bool = False
    OCR_USE_ANGLE_CLS: bool = True
    OCR_DET_DB_BOX_THRESH: float = 0.3
    OCR_CPU_THREADS: int = 4

    # Number of PaddleOCR engines kept in a pool per language. Pages of a PDF
    # are dispatched across these engines in parallel via asyncio.gather.
    # Each engine uses ~200-500MB RAM, so memory cost ≈ workers × ~400MB.
    # Each engine is given OCR_CPU_THREADS // OCR_PARALLEL_WORKERS intra-op
    # threads so the total CPU budget stays equal to OCR_CPU_THREADS.
    OCR_PARALLEL_WORKERS: int = Field(default=4, ge=1, le=32)

    # Run preprocessor.preprocess() (deskew + denoise + CLAHE) before OCR.
    # Helps phone-scanned PDFs (skewed, uneven lighting, noisy). Clean
    # pages skip the heavy steps via an internal "looks clean" heuristic.
    # Adds 100-500ms per page when applied; turn off if your inputs are
    # already clean digital rasters.
    OCR_ENABLE_PREPROCESS: bool = True

    OCR_MAX_FILE_SIZE_MB: int = 50
    OCR_MAX_PAGES: int = 100
    OCR_URL_DOWNLOAD_TIMEOUT: int = 30
    OCR_MIN_TEXT_LENGTH: int = 50
    OCR_PDF_DPI: int = 300

    # Default = mobile_det. Server_det gives slightly higher recall on small/dense
    # text but inference memory peak >8GB on Rosetta-emulated linux/amd64 → OOMKill.
    # Switch to "PP-OCRv5_server_det" only on native amd64 host with ≥16GB container limit.
    OCR_TEXT_DETECTION_MODEL: str = "PP-OCRv5_mobile_det"

    # Default = mobile_rec. PaddleOCR auto-selects PP-OCRv5_server_rec when this
    # is unset, which is ~3-4x slower on CPU and uses ~300MB more per engine.
    # Switch to "PP-OCRv5_server_rec" only on native amd64 host with ≥16GB RAM
    # and when accuracy matters more than throughput (e.g. small/dense text).
    OCR_TEXT_RECOGNITION_MODEL: str = "PP-OCRv5_mobile_rec"
    OCR_USE_DOC_ORIENTATION_CLASSIFY: bool = True
    # UVDoc unwarping default off: memory peak >8GB on Rosetta-emulated linux/amd64.
    # Enable (=true) on native amd64 host with ≥16GB RAM, or for phone-photo scans
    # of curved/folded documents where benefit > memory cost.
    OCR_USE_DOC_UNWARPING: bool = False
    OCR_UNCLEAR_THRESHOLD: float = 0.5

    LOG_LEVEL: str = "INFO"

    @field_validator("OCR_API_KEY")
    @classmethod
    def _key_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("OCR_API_KEY must not be blank")
        return v

    @property
    def max_file_bytes(self) -> int:
        return self.OCR_MAX_FILE_SIZE_MB * 1024 * 1024


settings = Settings()

# Tuning knobs that PaddlePaddle reads from the process environment.
# Must be set BEFORE paddle is imported (which happens lazily inside services).
#
# FLAGS_use_mkldnn: belt-and-suspenders. PaddleOCR 3.x PIR path actually
# honors the `enable_mkldnn` kwarg passed to PaddleOCR(), not this env var,
# but we still pin it here in case the legacy executor is taken.
os.environ["FLAGS_use_mkldnn"] = "1" if settings.OCR_ENABLE_MKLDNN else "0"

# Per-engine thread budget. With pool of N engines, OMP/MKL must NOT see the
# full OCR_CPU_THREADS or each parallel call grabs the whole budget → wild
# oversubscription (N × OCR_CPU_THREADS total threads). Match the per-engine
# `cpu_threads` kwarg we pass to PaddleOCR() so total stays at OCR_CPU_THREADS.
# Direct assignment (not setdefault) overrides any stale host/Dockerfile env.
_per_engine_threads = max(
    1, settings.OCR_CPU_THREADS // settings.OCR_PARALLEL_WORKERS
)
os.environ["OMP_NUM_THREADS"] = str(_per_engine_threads)
os.environ["MKL_NUM_THREADS"] = str(_per_engine_threads)

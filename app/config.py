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

    OCR_MAX_FILE_SIZE_MB: int = 50
    OCR_MAX_PAGES: int = 100
    OCR_URL_DOWNLOAD_TIMEOUT: int = 30
    OCR_MIN_TEXT_LENGTH: int = 50
    OCR_PDF_DPI: int = 300

    # Default = mobile_det. Server_det gives slightly higher recall on small/dense
    # text but inference memory peak >8GB on Rosetta-emulated linux/amd64 → OOMKill.
    # Switch to "PP-OCRv5_server_det" only on native amd64 host with ≥16GB container limit.
    OCR_TEXT_DETECTION_MODEL: str = "PP-OCRv5_mobile_det"
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
os.environ.setdefault("FLAGS_use_mkldnn", "1" if settings.OCR_ENABLE_MKLDNN else "0")
os.environ.setdefault("OMP_NUM_THREADS", str(settings.OCR_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(settings.OCR_CPU_THREADS))

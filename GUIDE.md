# CLAUDE.md — OCR Gateway Project Guide

> Panduan ini digunakan oleh Claude CLI (Claude Code) untuk memahami
> arsitektur, konvensi, dan cara kerja project OCR Gateway.
> Baca seluruh file ini sebelum membuat perubahan apapun.

---

## Project Overview

**Nama:** `ocr-gateway`
**Deskripsi:** Self-hosted OCR microservice berbasis FastAPI + PaddleOCR.
Menerima dokumen via file upload atau URL (presigned URL dari object storage
seperti RustFS/MinIO/S3), mengekstrak teks, dan mengembalikan hasil dalam
format yang siap digunakan oleh RAG pipeline.

**Tujuan utama:**
- Expose REST API untuk OCR dokumen PDF dan image
- Support input via file upload (multipart) atau URL
- Auth via Bearer Token
- Auto-detect PDF type (text-based vs scan)
- Swagger/OpenAPI docs tersedia di `/docs`
- Deploy via Docker, jalan di CPU tanpa GPU

---

## Tech Stack

```
Runtime:      Python 3.11+
Framework:    FastAPI
OCR Engine:   PaddleOCR v3 (PP-OCRv5)
PDF Parser:   PyMuPDF (fitz) — untuk detect text layer + convert page ke image
Image Proc:   OpenCV, Pillow
Validation:   Pydantic v2
API Docs:     Swagger UI (built-in FastAPI) di /docs
              ReDoc di /redoc
Auth:         Bearer Token (static, via environment variable)
Deploy:       Docker (CPU only, tidak butuh GPU)
```

---

## Directory Structure

```
ocr-gateway/
├── app/
│   ├── main.py                  # FastAPI app, router registration, lifespan
│   ├── config.py                # Settings via pydantic-settings + .env
│   ├── dependencies.py          # Auth dependency (Bearer Token verification)
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── services/
│   │   ├── detector.py          # Detect apakah PDF text-based atau scan
│   │   ├── pdf_extractor.py     # Extract text dari PDF digital (tanpa OCR)
│   │   ├── preprocessor.py      # Image preprocessing sebelum OCR
│   │   └── ocr_service.py       # PaddleOCR wrapper (singleton)
│   └── routers/
│       ├── ocr.py               # POST /extract — endpoint utama
│       └── health.py            # GET /health — health check
├── tests/
│   ├── conftest.py
│   ├── test_ocr.py
│   └── test_auth.py
├── sample_docs/                 # Dokumen sample untuk testing
│   ├── sample-text-pdf.pdf      # PDF digital (ada text layer)
│   └── sample-scan.jpg          # Contoh scan dokumen
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

```env
# Auth
OCR_API_KEY=your-secret-bearer-token-here   # WAJIB — Bearer token untuk auth

# PaddleOCR
OCR_LANGUAGE=en                              # en untuk latin/Indonesia, ch untuk Chinese
OCR_USE_GPU=false                            # false untuk CPU deployment
OCR_ENABLE_MKLDNN=true                       # WAJIB true untuk CPU performance
OCR_USE_ANGLE_CLS=true                       # Deteksi rotasi dokumen
OCR_DET_DB_BOX_THRESH=0.3                   # Threshold deteksi text box
OCR_CPU_THREADS=4                            # Jumlah thread CPU untuk PaddleOCR

# Processing
OCR_MAX_FILE_SIZE_MB=50                      # Maksimum ukuran file (MB)
OCR_MAX_PAGES=100                            # Maksimum halaman per dokumen
OCR_URL_DOWNLOAD_TIMEOUT=30                  # Timeout download dari URL (detik)
OCR_MIN_TEXT_LENGTH=50                       # Minimum karakter untuk anggap PDF punya text layer

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1                                    # Untuk CPU, 1 worker sudah cukup
LOG_LEVEL=info
```

---

## Authentication

**Semua endpoint (kecuali `/health` dan `/docs`) wajib menggunakan Bearer Token.**

### Request Header

```
Authorization: Bearer your-secret-bearer-token-here
```

### Dependency Implementation

```python
# app/dependencies.py
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.config import settings

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.OCR_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
```

### Endpoint yang TIDAK butuh auth

```
GET  /health          — health check, public
GET  /docs            — Swagger UI, public
GET  /redoc           — ReDoc, public
GET  /openapi.json    — OpenAPI schema, public
```

### Endpoint yang WAJIB auth

```
POST /extract         — OCR endpoint utama
GET  /formats         — list format yang didukung
```

---

## API Endpoints

### POST /extract — Endpoint Utama OCR

**Auth:** Bearer Token required

**Input:** `multipart/form-data` atau `application/json`

**Parameter:**

| Parameter       | Type            | Required | Default    | Description |
|----------------|-----------------|----------|------------|-------------|
| `file`         | UploadFile      | No*      | -          | File upload (PDF, JPG, PNG, TIFF) |
| `url`          | string          | No*      | -          | URL file dari object storage |
| `output_format`| string (enum)   | No       | `text`     | Format output: `text`, `markdown`, `json` |
| `language`     | string          | No       | `en`       | Bahasa: `en` (Latin/Indonesia), `ch` |
| `pages`        | string          | No       | `all`      | Halaman: `all`, `1`, `1-5`, `1,3,5` |

> *Salah satu dari `file` atau `url` WAJIB diisi. Tidak boleh keduanya sekaligus.

**Output Format Options:**

```
text     → Plain text, halaman dipisahkan dengan \n\n---\n\n
markdown → Struktur preserved (heading, tabel, list)
json     → Structured dengan metadata per halaman
```

**Response Success (200):**

```json
{
  "success": true,
  "engine": "paddleocr",
  "method": "ocr",
  "pages_processed": 3,
  "total_pages": 3,
  "output_format": "json",
  "result": {
    "pages": [
      {
        "page": 1,
        "text": "STANDARD OPERATING PROCEDURE\nPT Maju Bersama Indonesia...",
        "confidence": 0.94,
        "word_count": 245,
        "has_table": false,
        "has_unclear": false
      }
    ],
    "full_text": "STANDARD OPERATING PROCEDURE\n...",
    "metadata": {
      "file_name": "SOP-HR-001.pdf",
      "file_size_bytes": 204800,
      "processing_time_ms": 3420,
      "source": "upload"
    }
  }
}
```

**Response Error:**

```json
{
  "success": false,
  "error": "FILE_TOO_LARGE",
  "detail": "File size 75MB exceeds maximum allowed 50MB",
  "status_code": 413
}
```

**Error Codes:**

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `MISSING_INPUT` | 400 | Tidak ada file atau url |
| `BOTH_INPUT` | 400 | Dua-duanya diisi |
| `UNSUPPORTED_FORMAT` | 415 | Format file tidak didukung |
| `FILE_TOO_LARGE` | 413 | File melebihi batas maksimum |
| `TOO_MANY_PAGES` | 400 | Halaman melebihi batas |
| `URL_UNREACHABLE` | 422 | URL tidak bisa diakses |
| `URL_TIMEOUT` | 408 | Timeout saat download dari URL |
| `OCR_FAILED` | 500 | PaddleOCR error |
| `INVALID_PDF` | 422 | File PDF corrupt atau invalid |
| `UNAUTHORIZED` | 401 | Bearer token salah atau tidak ada |

---

### GET /health — Health Check

**Auth:** Tidak diperlukan

**Response (200):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "engine": "paddleocr",
  "engine_version": "3.0.0",
  "gpu_available": false,
  "mkldnn_enabled": true,
  "supported_formats": ["pdf", "jpg", "jpeg", "png", "tiff", "bmp", "webp"],
  "max_file_size_mb": 50,
  "max_pages": 100
}
```

---

### GET /formats — List Format yang Didukung

**Auth:** Bearer Token required

**Response (200):**

```json
{
  "input_formats": {
    "documents": ["pdf"],
    "images": ["jpg", "jpeg", "png", "tiff", "bmp", "webp"]
  },
  "output_formats": {
    "text": "Plain text, pages separated by ---",
    "markdown": "Markdown with preserved structure",
    "json": "Structured JSON with per-page metadata"
  },
  "input_methods": {
    "upload": "multipart/form-data file upload",
    "url": "HTTP/HTTPS URL to file (presigned URLs supported)"
  }
}
```

---

## Core Services

### `detector.py` — PDF Type Detection

```python
# Detect apakah PDF punya text layer atau scan
# Logic:
# 1. Buka PDF dengan PyMuPDF
# 2. Extract text dari halaman pertama
# 3. Kalau len(text.strip()) > MIN_TEXT_LENGTH → text-based
# 4. Kalau kosong atau sangat sedikit → scan, butuh OCR

# PENTING:
# - Jangan anggap PDF "text-based" hanya karena ada sedikit teks
#   (bisa jadi watermark atau header saja)
# - MIN_TEXT_LENGTH default: 50 karakter per halaman
# - Test di beberapa halaman, bukan hanya halaman pertama
```

### `pdf_extractor.py` — Text Extraction (No OCR)

```python
# Untuk PDF yang sudah punya text layer
# Gunakan PyMuPDF (fitz) — lebih baik dari pdfminer untuk layout

# Rules:
# - Preserve paragraph breaks
# - Handle multi-column layout
# - Extract tabel sebagai plain text (per cell)
# - Jangan strip whitespace yang meaningful
# - Return per-page result untuk metadata
```

### `preprocessor.py` — Image Preprocessing

```python
# Preprocessing sebelum PaddleOCR untuk improve akurasi scan gadget
# Pipeline (urutan penting):
# 1. Convert ke grayscale
# 2. Deskew (luruskan dokumen miring) — pakai OpenCV
# 3. Denoise — cv2.fastNlMeansDenoising
# 4. Contrast enhancement — CLAHE
# 5. Binarization — Otsu thresholding
# 6. Resize kalau terlalu kecil (min 1000px lebar)

# JANGAN over-process:
# - Kalau dokumen sudah bagus, skip aggressive preprocessing
# - Preprocessing berlebihan bisa MERUSAK akurasi
# - Implement quality check sebelum decide preprocessing level
```

### `ocr_service.py` — PaddleOCR Wrapper

```python
# Singleton pattern — inisialisasi PaddleOCR sekali saat startup
# Jangan inisialisasi per-request (sangat lambat)

# Konfigurasi wajib untuk CPU:
# use_gpu=False
# enable_mkldnn=True   ← CRITICAL untuk performance
# use_angle_cls=True   ← Untuk dokumen yang mungkin dirotasi
# lang='en'            ← Untuk Bahasa Indonesia (latin script)

# Per-halaman processing:
# - PDF → convert setiap halaman ke image (via PyMuPDF)
# - Image → langsung ke PaddleOCR
# - Confidence score per halaman
# - Flag [UNCLEAR] untuk area yang confidence < threshold

# PENTING untuk dokumen hukum:
# - Jangan auto-correct teks
# - Jangan interpret — extract exactly as-is
# - Kalau confidence < 0.5 → mark sebagai [UNCLEAR], jangan guess
```

---

## Swagger / OpenAPI Documentation

FastAPI generate Swagger otomatis. Konfigurasi di `main.py`:

```python
app = FastAPI(
    title="OCR Gateway",
    description="""
## OCR Gateway API

Self-hosted OCR service powered by PaddleOCR.
Extract text from PDF documents and images via file upload or URL.

### Authentication
All endpoints (except `/health` and docs) require Bearer Token authentication.

Include in request header:
```
Authorization: Bearer your-token-here
```

### Supported Input
- **File Upload**: multipart/form-data
- **URL**: Presigned URL from RustFS, MinIO, S3, or any HTTP URL

### Supported Formats
- Documents: PDF
- Images: JPG, PNG, TIFF, BMP, WebP
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "OCR", "description": "Text extraction endpoints"},
        {"name": "System", "description": "Health check and system info"},
    ]
)
```

**Swagger UI tersedia di:** `http://localhost:8000/docs`
**ReDoc tersedia di:** `http://localhost:8000/redoc`

---

## Request Examples

### Upload File

```bash
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer your-token-here" \
  -F "file=@/path/to/document.pdf" \
  -F "output_format=json" \
  -F "language=en"
```

### URL dari Object Storage (Presigned URL)

```bash
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://rustfs.tabalongkab.go.id/bucket/doc.pdf?X-Amz-Signature=...",
    "output_format": "markdown",
    "language": "en",
    "pages": "1-10"
  }'
```

### Dari Laravel (PHP)

```php
// Upload file
$response = Http::withHeaders([
    'Authorization' => 'Bearer ' . config('services.ocr.key'),
])->attach(
    'file', file_get_contents($filePath), basename($filePath)
)->post(config('services.ocr.url') . '/extract', [
    'output_format' => 'json',
    'language' => 'en',
]);

// Via presigned URL
$presignedUrl = Storage::temporaryUrl($path, now()->addMinutes(5));

$response = Http::withHeaders([
    'Authorization' => 'Bearer ' . config('services.ocr.key'),
])->post(config('services.ocr.url') . '/extract', [
    'url' => $presignedUrl,
    'output_format' => 'json',
]);

$result = $response->json();
$text = $result['result']['full_text'];
```

---

## Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim

# System dependencies untuk PaddleOCR dan OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR models saat build
# Sehingga tidak download saat runtime (lebih cepat startup)
RUN python -c "from paddleocr import PaddleOCR; \
    PaddleOCR(use_gpu=False, lang='en', show_log=False)"

COPY app/ ./app/
COPY .env.example .env

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  ocr-gateway:
    build: .
    container_name: ocr-gateway
    ports:
      - "8000:8000"
    environment:
      - OCR_API_KEY=${OCR_API_KEY}
      - OCR_LANGUAGE=en
      - OCR_USE_GPU=false
      - OCR_ENABLE_MKLDNN=true
      - OCR_USE_ANGLE_CLS=true
      - OCR_MAX_FILE_SIZE_MB=50
      - OCR_MAX_PAGES=100
      - OCR_CPU_THREADS=4
      - LOG_LEVEL=info
    volumes:
      - ocr-temp:/tmp/ocr-temp   # Temp storage untuk file processing
    restart: unless-stopped
    networks:
      - dokploy-network           # Shared network dengan Laravel project

volumes:
  ocr-temp:

networks:
  dokploy-network:
    external: true
```

---

## Coding Standards

### Python Style

- **Python 3.11+** — gunakan fitur modern (match statement, type hints)
- **Type hints WAJIB** untuk semua function signature
- **Pydantic v2** untuk semua schema — jangan dict manual
- **Async/await** untuk I/O operations (download URL, file read)
- **Singleton** untuk PaddleOCR instance — inisialisasi di lifespan, bukan per-request
- **Dependency Injection** via FastAPI `Depends()`

### Error Handling

```python
# Selalu gunakan custom exception dengan error code yang jelas
class OcrGatewayException(Exception):
    def __init__(self, error_code: str, detail: str, status_code: int):
        self.error_code = error_code
        self.detail = detail
        self.status_code = status_code

# Contoh
raise OcrGatewayException(
    error_code="FILE_TOO_LARGE",
    detail=f"File size {size_mb}MB exceeds maximum {max_mb}MB",
    status_code=413
)
```

### Logging

```python
import logging
logger = logging.getLogger(__name__)

# Log setiap request dengan timing
logger.info(f"OCR request: method={method}, pages={pages}, time={ms}ms")

# Log error dengan context
logger.error(f"OCR failed: file={filename}, error={str(e)}", exc_info=True)

# Jangan log content dokumen — bisa berisi data sensitif
```

### Security

- Jangan log content dokumen atau teks hasil OCR
- Hapus file temporary setelah processing (gunakan `finally` block)
- Validate file type dari content, bukan hanya extension
- Batasi ukuran file dan jumlah halaman
- URL validation: hanya allow HTTP/HTTPS, reject file:// atau internal IP
  kecuali yang explicitly diwhitelist

---

## Processing Flow

```
Request masuk (file upload atau URL)
          ↓
Auth check (Bearer Token)
          ↓
Input validation (ukuran, format, URL accessibility)
          ↓
Download file (kalau input URL)
          ↓
Detect PDF type
     ↓              ↓
Text-based PDF    Scan / Image PDF
     ↓              ↓
PyMuPDF extract   Preprocessing (deskew, denoise)
     ↓              ↓
                  PaddleOCR per halaman
                  ↓
Format output (text / markdown / json)
          ↓
Cleanup temp files
          ↓
Return response
```

---

## Performance Notes

### Startup

- PaddleOCR model download terjadi saat Docker build (bukan runtime)
- Cold start pertama setelah container up: ~5-10 detik (model loading)
- Request pertama mungkin lebih lambat karena warm-up

### Per-Request

```
PDF text-based (10 halaman):   ~500ms  (PyMuPDF, no OCR)
PDF scan (1 halaman):          ~6-10s  (preprocessing + PaddleOCR)
PDF scan (10 halaman):         ~60-90s (sequential per halaman)
Image (1 file):                ~3-5s   (preprocessing + PaddleOCR)
```

### Optimasi

- `enable_mkldnn=True` — wajib, reduce time 3-5x di CPU
- `cpu_threads=4` — sesuaikan dengan jumlah core server
- Process halaman secara sequential (bukan parallel) untuk CPU
- Parallel processing hanya kalau ada multiple worker

---

## Testing

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest httpx pytest-asyncio

# Run semua test
pytest tests/ -v

# Test dengan file sample
pytest tests/test_ocr.py -v -k "test_upload_pdf"

# Test auth
pytest tests/test_auth.py -v
```

### Test Cases Wajib

```python
# test_auth.py
- test_request_without_token()      → expect 401
- test_request_with_wrong_token()   → expect 401
- test_request_with_valid_token()   → expect 200

# test_ocr.py
- test_upload_text_pdf()            → should use PyMuPDF, fast
- test_upload_scan_pdf()            → should use PaddleOCR
- test_upload_image_jpg()           → should use PaddleOCR
- test_url_presigned()              → download + OCR
- test_url_invalid()                → expect 422
- test_file_too_large()             → expect 413
- test_unsupported_format()         → expect 415
- test_no_input()                   → expect 400
- test_both_inputs()                → expect 400
- test_output_format_text()         → verify format
- test_output_format_json()         → verify structure
- test_output_format_markdown()     → verify format
```

---

## What NOT To Do

```
❌ Jangan inisialisasi PaddleOCR per-request — sangat lambat
❌ Jangan simpan file yang di-upload secara permanen — hapus setelah proses
❌ Jangan log content/teks hasil OCR — data sensitif
❌ Jangan skip auth check di endpoint manapun kecuali /health dan /docs
❌ Jangan allow URL ke internal network (SSRF) — validate URL
❌ Jangan gunakan use_gpu=True kalau tidak ada GPU — akan error
❌ Jangan skip enable_mkldnn=True — performance akan sangat buruk
❌ Jangan process PDF lebih dari max_pages tanpa reject — bisa OOM
❌ Jangan "guess" teks yang tidak terbaca — mark sebagai [UNCLEAR]
❌ Jangan auto-correct ejaan dalam hasil OCR — preserve as-is
```

---

## Integration dengan laravel-rag-mcp

OCR Gateway ini adalah **dependency dari `laravel-rag-mcp`**.

```
laravel-rag-mcp/.env:
OCR_GATEWAY_URL=http://ocr-gateway:8000
OCR_GATEWAY_KEY=your-bearer-token

# Di docker-compose laravel-rag-mcp:
# Pastikan ocr-gateway ada di network yang sama
# (dokploy-network)
```

```php
// app/Services/Ocr/OcrGatewayClient.php di laravel-rag-mcp
class OcrGatewayClient
{
    public function extractFromUpload(string $filePath): OcrResult { }
    public function extractFromUrl(string $presignedUrl): OcrResult { }
}
```

---

## Contact & Context

**Developer:** Hafiz Nugraha
**Email:** hafiznugrahaindonesia@gmail.com
**GitHub:** github.com/hafiznugrahadev

**Tujuan project:** OCR microservice untuk `laravel-rag-mcp`
**Target deployment:** Self-hosted via Docker, CPU only
**Primary use case:** Dokumen hukum scan gadget (miring, shadow, bervariasi)
**Language target:** Bahasa Indonesia (latin script) + English mixed
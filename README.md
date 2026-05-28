# OCR Gateway

Self-hosted OCR microservice powered by **PaddleOCR 3.x (PP-OCRv5)**, exposed as a clean HTTP API behind Bearer-token auth. Built to be consumed by `laravel-rag-mcp` (or any HTTP client) over a shared Docker network.

Tuned for Indonesian/English **legal documents** — including phone-photographed scans (skewed, rotated, perspective-distorted).

---

## Features

- **POST /extract** — single endpoint for both file upload and remote URL (S3/MinIO/RustFS presigned URLs supported)
- **Auto PDF type detection** — digital PDFs use the text layer (no OCR, ~50 ms); scanned PDFs go through OCR
- **Three output formats** — `json`, `text`, `markdown`
- **Page selector** — `all`, `1`, `1-5`, `1,3,5`, `1-3,7-10`
- **`[UNCLEAR]` markers** for low-confidence lines (no guessing)
- **Doc orientation auto-correct** — handles 0/90/180/270 rotation from phone photos
- **Bearer-token auth** + structured error responses with stable codes
- **CPU and GPU images** — same code, two Dockerfiles; toggle via env on the GPU host
- **Swagger UI** at `/docs`

---

## Quick Start (CPU, Docker)

```bash
git clone <this-repo> ocr-gateway && cd ocr-gateway
cp .env.example .env
# edit .env: set OCR_API_KEY=<your-secret>

docker build -t ocr-gateway:cpu .

docker run -d --name ocr-gateway \
  -p 8000:8000 \
  --env-file .env \
  --memory=8g --memory-swap=8g \
  -v ocr_models:/home/appuser/.paddlex \
  --restart unless-stopped \
  ocr-gateway:cpu
```

> On macOS (Apple Silicon) add `--platform linux/amd64` to both `docker build` and `docker run`. Paddle has no native arm64 wheel that works reliably; Rosetta emulation on amd64 is the proven path.

Test:

```bash
curl -s localhost:8000/health
# → {"status":"healthy","engine":"paddleocr","engine_version":"3.5.0",...}

curl -s -X POST localhost:8000/extract \
  -H "Authorization: Bearer <your-secret>" \
  -F file=@sample.png | jq .

# Or via remote URL:
curl -s -X POST localhost:8000/extract \
  -H "Authorization: Bearer <your-secret>" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/document.pdf"}' | jq .
```

Open <http://localhost:8000/docs> in a browser for interactive Swagger UI.

---

## GPU Quick Start (NVIDIA host)

Prerequisites on the host:
- NVIDIA driver (`nvidia-smi` works)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Sanity check: `docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 nvidia-smi`

```bash
docker build -f Dockerfile.gpu -t ocr-gateway:gpu .
```

Edit `.env` on the GPU host (re-enable the heavy quality knobs that VRAM lets you afford):

```env
OCR_API_KEY=<your-secret>
OCR_USE_GPU=true
OCR_TEXT_DETECTION_MODEL=PP-OCRv5_server_det
OCR_USE_DOC_UNWARPING=true
OCR_PDF_DPI=300
```

Run with `--gpus all`:

```bash
docker run -d --name ocr-gateway \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -v ocr_models:/home/appuser/.paddlex \
  --restart unless-stopped \
  ocr-gateway:gpu
```

> The `OCR_USE_GPU=true` switch only works **inside the GPU image**. The CPU image's paddlepaddle wheel has no CUDA bindings; setting the flag there will not magically enable GPU.

---

## API

### `GET /health` (no auth)

Liveness + engine info.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "engine": "paddleocr",
  "engine_version": "3.5.0",
  "gpu_available": false,
  "mkldnn_enabled": false,
  "supported_formats": ["pdf","jpg","jpeg","png","tiff","bmp","webp"],
  "max_file_size_mb": 100,
  "max_pages": 1000
}
```

Returns `503 {"status":"warming"}` during the first ~5-10 seconds while models load.

### `POST /extract` (Bearer required)

Either upload a file or pass a URL. Use exactly one.

**Multipart form (`multipart/form-data`):**

| Field | Required | Default | Description |
|---|---|---|---|
| `file` | one of | — | PDF or image upload |
| `url` | one of | — | HTTP/HTTPS URL to a file (presigned URLs OK) |
| `language` | no | `OCR_LANGUAGE` | `en`, `ch`, `latin`, etc. |
| `output_format` | no | `json` | `json`, `text`, or `markdown` |
| `pages` | no | `all` | `all`, `1`, `1-5`, `1,3,5`, `1-3,7-10` |

**JSON body** (alternative — convenient when you only have a URL):

```json
{
  "url": "https://rustfs.example.com/bucket/doc.pdf?presigned=...",
  "language": "en",
  "output_format": "json",
  "pages": "all"
}
```

**Successful response** (HTTP 200):

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
        "text": "STANDARD OPERATING PROCEDURE\nPT Maju Bersama...",
        "confidence": 0.94,
        "word_count": 245,
        "has_table": false,
        "has_unclear": false
      }
    ],
    "full_text": "STANDARD OPERATING PROCEDURE\n...",
    "metadata": {
      "file_name": "doc.pdf",
      "file_size_bytes": 204800,
      "processing_time_ms": 3420,
      "source": "upload"
    }
  }
}
```

`method` is either:
- `text-layer` — PDF had a text layer; extracted with PyMuPDF, no OCR (super fast)
- `ocr` — actual OCR was run

**Error response** (any non-2xx):

```json
{
  "success": false,
  "error": "FILE_TOO_LARGE",
  "detail": "File size 75MB exceeds maximum 50MB",
  "status_code": 413
}
```

Stable error codes: `MISSING_INPUT`, `BOTH_INPUT`, `UNSUPPORTED_FORMAT`, `FILE_TOO_LARGE`, `TOO_MANY_PAGES`, `INVALID_PAGES`, `INVALID_PDF`, `INVALID_URL`, `URL_UNREACHABLE`, `URL_TIMEOUT`, `INVALID_OUTPUT_FORMAT`, `OCR_FAILED`, `UNAUTHORIZED`, `VALIDATION_ERROR`.

### `GET /formats` (Bearer required)

Lists supported input/output formats. Useful for client introspection.

### `GET /docs` and `/redoc` (no auth)

Swagger UI / ReDoc.

---

## Examples

### Image upload (curl)

```bash
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer $OCR_API_KEY" \
  -F file=@receipt.jpg \
  -F output_format=text
```

### PDF, only pages 1-5, as Markdown

```bash
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer $OCR_API_KEY" \
  -F file=@doc.pdf \
  -F pages=1-5 \
  -F output_format=markdown
```

### Remote URL via JSON body

```bash
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer $OCR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://rustfs.example.com/bucket/contract.pdf",
    "output_format": "json",
    "pages": "all"
  }'
```

### Laravel (queue worker)

OCR jobs may take minutes for large scanned PDFs. Run them inside a queue worker, not a sync HTTP request:

```php
// app/Jobs/OcrDocument.php
public int $timeout = 1800; // 30 minutes

public function handle(): void
{
    $response = Http::withHeaders([
        'Authorization' => 'Bearer ' . config('services.ocr.key'),
    ])->timeout(1800)->acceptJson()->post(
        config('services.ocr.url') . '/extract',
        [
            'url' => $this->presignedUrl,
            'output_format' => 'json',
            'language' => 'en',
        ]
    );

    if (! $response->successful()) {
        throw new RuntimeException(
            "OCR failed: {$response->json('error')} - {$response->json('detail')}"
        );
    }

    $text = $response->json('result.full_text');
    // ... store / index / etc.
}
```

In `config/queue.php` bump `retry_after` to ≥ `1900`. In Horizon, set `timeout: 1900` on the relevant supervisor.

---

## Configuration

All settings come from environment variables (loaded from `.env` via `--env-file`).

| Variable | Default | Purpose |
|---|---|---|
| `OCR_API_KEY` | — (**required**) | Bearer token clients must present |
| `OCR_LANGUAGE` | `en` | PaddleOCR language code (default when client omits `language`) |
| `OCR_USE_GPU` | `false` | Set `true` only on GPU image |
| `OCR_ENABLE_MKLDNN` | `false` | Disabled — Paddle 3.x PIR + OneDNN has unimplemented op coverage |
| `OCR_USE_ANGLE_CLS` | `true` | Per-line orientation classifier |
| `OCR_USE_DOC_ORIENTATION_CLASSIFY` | `true` | Detects 0/90/180/270 rotation (phone photos) |
| `OCR_USE_DOC_UNWARPING` | `false` | UVDoc perspective-correction. ~2 GB peak memory; enable on GPU or hosts with ≥16 GB |
| `OCR_TEXT_DETECTION_MODEL` | `PP-OCRv5_mobile_det` | Use `PP-OCRv5_server_det` on GPU/large-RAM hosts for slightly better recall |
| `OCR_TEXT_RECOGNITION_MODEL` | `PP-OCRv5_server_rec` | Recognition model. See [Recognition models](#recognition-models) for trade-offs |
| `OCR_DET_DB_BOX_THRESH` | `0.3` | Detection threshold |
| `OCR_CPU_THREADS` | `4` | OMP/MKL thread count |
| `OCR_PDF_DPI` | `200` | Rasterization DPI for scanned PDFs (300 = sharper, ~2× memory) |
| `OCR_MAX_FILE_SIZE_MB` | `50` | Hard cap on upload + remote download |
| `OCR_MAX_PAGES` | `100` | PDFs over this return `TOO_MANY_PAGES` |
| `OCR_URL_DOWNLOAD_TIMEOUT` | `30` | Seconds to wait when fetching from URL |
| `OCR_MIN_TEXT_LENGTH` | `50` | Threshold (chars/page) for "is this PDF text-based?" |
| `OCR_UNCLEAR_THRESHOLD` | `0.5` | Lines below this confidence become `[UNCLEAR]` |
| `LOG_LEVEL` | `INFO` | |

---

## Recognition models

`OCR_TEXT_RECOGNITION_MODEL` selects which PaddleOCR recognition model converts each detected text box into a string. The detection model (`OCR_TEXT_DETECTION_MODEL`) finds the boxes; the recognition model reads them. They are independent — you can mix any detection model with any recognition model.

| Model | Language scope | Speed (CPU) | RAM / engine | Inter-word spaces | When to use |
|---|---|---|---|---|---|
| `PP-OCRv5_server_rec` *(default)* | CN + EN + JP + KR + Latin | baseline | ~700 MB | **Reliable** | Indonesian / English legal docs, especially with ALL-CAPS headings (`KEPUTUSAN BUPATI TABALONG`) |
| `PP-OCRv5_mobile_rec` | CN + EN + JP + KR + Latin | ~3–4× faster | ~400 MB | **Drops them** on dense bold ALL-CAPS lines → `BUPATITABALONG` | Throughput-first pipelines on clean lowercase body text only |
| `latin_PP-OCRv5_mobile_rec` | Latin scripts only (EN, ID, FR, DE, PT, ES, …) | ~3× faster than server_rec | ~300 MB | Reliable | Pure Indonesian/English workloads that want mobile speed without the space-dropping issue |
| `en_PP-OCRv5_mobile_rec` | English only | ~3× faster than server_rec | ~300 MB | Reliable | English-only inputs (no Indonesian / no other Latin scripts) |

**Recommendation per workload:**

- **Indonesian government / legal documents (the original target of this gateway):** stay on `PP-OCRv5_server_rec`. ALL-CAPS headings are the norm and you cannot afford `BUPATITABALONG`-class concatenation.
- **High-throughput pipelines on clean printed body text:** `latin_PP-OCRv5_mobile_rec` gives most of the speed of `mobile_rec` without the inter-word-space bug, as long as your inputs are Latin-script only.
- **Mixed CJK + Latin content:** `PP-OCRv5_server_rec` is the only safe choice; `mobile_rec` has the same issue with dense Chinese/Japanese as it does with ALL-CAPS Latin.

**Memory note:** the gateway provisions `OCR_PARALLEL_WORKERS` recognition engines at startup (see `_ensure_pool` in `app/services/ocr_service.py`). Total pool RAM ≈ `OCR_PARALLEL_WORKERS × (RAM/engine from the table above)`. Pick the model **and** the worker count together for your host: e.g. `server_rec` × 4 workers ≈ 2.8 GB; `server_rec` × 16 workers ≈ 11.2 GB.

> Note: `engine=onnx` (RapidOCR backend) uses its own bundled PP-OCRv4 ONNX model and ignores `OCR_TEXT_RECOGNITION_MODEL`. This setting only affects the default `engine=paddle` path.

---

## Performance

Rough numbers for a typical 11-12 pt Indonesian/English legal document.

| Scenario | CPU (8-core, mobile_det, DPI 200) | GPU (GTX 1060+, server_det, DPI 300) |
|---|---|---|
| Clean image, 1 page | ~600 ms | ~50–100 ms |
| Digital PDF with text layer | ~50 ms | ~50 ms (no OCR) |
| Scanned PDF, 1 page | ~10–25 s | ~1–3 s |
| Scanned PDF, 50 pages | ~10–15 min | ~1–3 min |
| Scanned PDF, 100 pages | ~20–30 min | ~3–5 min |

Recognition (`PP-OCRv5_server_rec`) is always server-grade, regardless of detection model — text that gets detected is always read at high accuracy.

---

## Deploying alongside `laravel-rag-mcp` on Dokploy

The Laravel client expects:

```env
OCR_GATEWAY_URL=http://ocr-gateway:8000
OCR_GATEWAY_KEY=<same value as the server's OCR_API_KEY>
```

Steps:

1. Deploy this image as a Dokploy service named **`ocr-gateway`** (the hostname must match).
2. Attach the service to the **`dokploy-network`** so it shares a network with `laravel-rag-mcp`.
3. Set `OCR_API_KEY` server-side and `OCR_GATEWAY_KEY` client-side to the **same** value.
4. Mount a persistent volume on `/home/appuser/.paddlex` to avoid re-downloading model weights on every restart.

Verify from inside the Laravel container:

```bash
curl http://ocr-gateway:8000/health
```

---

## Project layout

```
ocr-gateway/
├── Dockerfile             # CPU build (python:3.11-slim + paddlepaddle CPU)
├── Dockerfile.gpu         # GPU build (CUDA 12.6 runtime + paddlepaddle-gpu)
├── requirements.txt       # CPU deps
├── requirements-gpu.txt   # GPU deps (paddlepaddle-gpu via Paddle's index)
├── .env.example
├── README.md              # this file
└── app/
    ├── main.py            # FastAPI app, lifespan, exception handlers
    ├── config.py          # pydantic-settings; reads .env
    ├── dependencies.py    # require_bearer auth dependency
    ├── routers/
    │   ├── extract.py     # POST /extract
    │   └── health.py      # GET /health, GET /formats
    ├── services/
    │   ├── detector.py        # PDF text-layer detection (PyMuPDF)
    │   ├── pdf_extractor.py   # text-layer extraction (no OCR)
    │   ├── pdf_rasterizer.py  # PDF page → PNG (PyMuPDF)
    │   ├── ocr_service.py     # PaddleOCR 3.x wrapper (singleton per lang)
    │   ├── url_fetcher.py     # remote URL download with size cap
    │   └── preprocessor.py    # legacy CV preprocessor (not in runtime path)
    ├── models/
    │   └── schemas.py     # Pydantic request/response schemas
    └── utils/
        ├── errors.py      # OcrGatewayError + structured error handler
        └── pages.py       # page selector parser
```

---

## Troubleshooting

### `OOMKilled` (exit code 137) during OCR

Container ran out of memory mid-inference. On Rosetta-emulated linux/amd64 (Apple Silicon Macs), peak memory for `server_det` + `UVDoc` + DPI 300 can exceed 8 GB.

Fix (any of these):
1. Lower `OCR_PDF_DPI` to `200`
2. Set `OCR_TEXT_DETECTION_MODEL=PP-OCRv5_mobile_det`
3. Set `OCR_USE_DOC_UNWARPING=false`
4. Bump `--memory=12g` (only if Docker Desktop has enough RAM allocated)
5. Run on a real GPU host (no Rosetta overhead)

The default `.env.example` ships with the safe combo.

### Container is stuck "warming" forever

Models are downloaded from Hugging Face on first request. If the host has no internet (corp proxy, air-gapped), download fails and `/health` stays at `warming`. Either:
- Allow outbound to `huggingface.co` and `bcebos.com`
- Pre-bake weights into the image at build time and mount as volume

### Hoppscotch / browser shows "Loading..." but server log shows `extract completed`

Some upstream proxy (Cloudflare, Nginx) timed out the open connection while the server was still working. The OCR result was produced but never delivered. **Solution:** call `/extract` from a server-to-server context (e.g. Laravel queue worker), not from a browser-proxied client. There is no client-side proxy timeout when both services share a Docker network.

### `[UNCLEAR]` appears in the output

A line was detected but recognition confidence fell below `OCR_UNCLEAR_THRESHOLD` (default `0.5`). The original recognized text is still preserved per-line in `result.pages[i].lines[j].text`; only the joined `full_text` substitutes `[UNCLEAR]`. Lower the threshold to `0.3` if you'd rather see noisy text than skipped tokens.

### `lang` and `ocr_version` will be ignored…

A harmless warning from PaddleOCR 3.x: when you pass an explicit model name (`OCR_TEXT_DETECTION_MODEL=...`), the language hint is overridden by the model itself. Recognition still works for the model's supported languages.

---

## License

MIT — same as the upstream PaddleOCR. Models are downloaded from Hugging Face under their respective licenses.

# ColorComic

Automatically colorize black-and-white comic and manga pages using deep learning. Upload a PDF, get back a fully colorized version — no API keys, no cloud services, everything runs locally on your machine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Fully automatic colorization** — no prompts, no reference images, no manual intervention
- **GPU accelerated** — 2-5 seconds per page on CUDA GPUs, with automatic CPU fallback
- **GPU detection** — analyze your hardware specs before choosing a device
- **Cross-page color consistency** — LAB color transfer keeps character/environment colors consistent across pages
- **Live preview** — side-by-side original vs. colorized comparison updates in real-time during processing
- **PDF in, PDF out** — upload a B&W comic PDF, download a colorized PDF
- **Zero cloud dependency** — everything runs locally, no API keys needed
- **Auto model download** — weights are downloaded automatically on first run (~140 MB)

## How It Works

```
Upload PDF
    |
    v
Extract pages at 300 DPI (PyMuPDF)
    |
    v
Denoise each page (FFDNet)
    |
    v
Colorize page 1 (manga-colorization-v2)
    |
    v
Store page 1 as color reference (LAB statistics)
    |
    v
For pages 2..N:
    Colorize with manga-colorization-v2
    Apply LAB color transfer using page 1 as reference
    |
    v
Reassemble into PDF with compression
    |
    v
Preview & Download
```

The colorization model is [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2), a U-Net architecture with an SEResNeXt encoder trained specifically on manga artwork. Cross-page consistency uses Reinhard LAB color transfer on the A and B (chrominance) channels, preserving the model's luminance decisions while aligning the color palette across all pages.

## Requirements

- **Python** 3.10+
- **PyTorch** 2.0+ (with CUDA for GPU acceleration)
- **~500 MB disk space** for model weights (downloaded automatically)
- **GPU (optional):** Any NVIDIA GPU with 2+ GB VRAM and CUDA support. An RTX 3060 or better is ideal. CPU mode works but is significantly slower (~30s/page vs ~3s/page).

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vikast908/ColorComic.git
cd ColorComic
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install PyTorch

**With GPU (NVIDIA CUDA):**

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and select your CUDA version, or:

```bash
# CUDA 12.8 (most recent NVIDIA drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**

```bash
pip install torch torchvision
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment (optional)

```bash
cp .env.example .env
```

Edit `.env` if you want to change defaults:

```env
SECRET_KEY=change-this-to-a-random-string
COLORCOMIC_DEVICE=auto       # auto | cpu | cuda
COLOR_TRANSFER_STRENGTH=0.7  # 0.0 = no transfer, 1.0 = full transfer
```

### 6. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. On first run, model weights (~140 MB) will be downloaded automatically from Google Drive.

## Usage

1. **Upload** — Drop a B&W comic/manga PDF onto the upload page
2. **Detect GPU** — Click "Detect GPU" to see your hardware specs and pick CPU or GPU
3. **Colorize** — Hit "Upload & Colorize" and watch the progress with live previews
4. **Download** — Get the colorized PDF or review individual pages

## Configuration

| Variable | Default | Description |
|---|---|---|
| `COLORCOMIC_DEVICE` | `auto` | Device for inference. `auto` picks GPU if available, otherwise CPU. Set to `cpu` or `cuda` to force a specific device. |
| `COLOR_TRANSFER_STRENGTH` | `0.7` | How strongly to align colors across pages (0.0–1.0). Lower values give the model more freedom per page; higher values enforce stricter consistency. |
| `SECRET_KEY` | random | Flask session secret. Set to a fixed string in production. |

## Project Structure

```
ColorComic/
├── app.py                    # Flask application & routes
├── config.py                 # Configuration (env vars, paths)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
│
├── core/                     # Core pipeline modules
│   ├── ml_colorizer.py       # Thread-safe model wrapper with OOM fallback
│   ├── color_consistency.py  # LAB color transfer for cross-page consistency
│   ├── model_downloader.py   # Auto-download weights from Google Drive
│   ├── pdf_handler.py        # PDF extraction & reassembly (PyMuPDF)
│   └── panel_detector.py     # Panel detection (available for future use)
│
├── models/
│   ├── schemas.py            # Pydantic data models (JobState, PanelRegion)
│   └── weights/              # Model weights (auto-downloaded, gitignored)
│       ├── generator.zip     # Main colorizer (~129 MB)
│       └── denoiser/
│           └── net_rgb.pth   # FFDNet denoiser (~3.4 MB)
│
├── vendor/
│   └── manga_colorization_v2/  # Vendored inference code
│       ├── colorizator.py      # Model orchestrator
│       ├── networks/           # Generator, Colorizer, SEResNeXt
│       ├── denoising/          # FFDNet denoiser
│       └── utils/              # Image preprocessing
│
├── templates/                # Jinja2 HTML templates
│   ├── base.html             # Layout with dark theme
│   ├── index.html            # Upload page with GPU detection
│   ├── processing.html       # Live progress with side-by-side preview
│   └── preview.html          # Page-by-page review & download
│
└── static/
    ├── css/style.css         # Dark theme stylesheet
    └── js/
        ├── app.js            # Global JS (theme, navigation)
        └── upload.js         # Upload logic & GPU detection
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Upload page |
| `POST` | `/upload` | Upload a PDF, returns `job_id` |
| `POST` | `/api/colorize/<job_id>` | Start colorization pipeline |
| `GET` | `/api/colorize/<job_id>/stream` | SSE stream of progress events |
| `GET` | `/api/preview/<job_id>/<page>` | Serve a colorized page image |
| `GET` | `/pages/<job_id>/<page>` | Serve an original B&W page image |
| `GET` | `/api/download/<job_id>` | Download the colorized PDF |
| `GET` | `/api/gpu-info` | GPU detection (name, VRAM, compute capability) |
| `GET` | `/api/status` | Model health check |
| `GET` | `/processing/<job_id>` | Processing page with live preview |
| `GET` | `/preview/<job_id>` | Review colorized pages |

## GPU Support

ColorComic auto-detects CUDA GPUs at startup. On the upload page, click **"Detect GPU"** to see:

- GPU name and model
- Total and free VRAM
- Compute capability and SM count
- CUDA toolkit version
- A recommendation (GPU or CPU based on available VRAM)

**Minimum:** Any NVIDIA GPU with 2 GB VRAM
**Recommended:** 4+ GB VRAM (RTX 3060, RTX 4070, etc.)
**CPU fallback:** Automatic if GPU runs out of memory mid-inference

If PyTorch was installed without CUDA support (`torch+cpu`), GPU will not be available regardless of hardware. Reinstall with the correct CUDA index URL (see [Installation](#3-install-pytorch)).

## Limitations

- **Manga-optimized:** The model is trained on manga/anime artwork. Western comics with heavy stippling or unique art styles may get lower-quality results.
- **Consistency is approximate:** LAB color transfer aligns global color distributions, not specific character elements. Two characters wearing different colors may shift slightly between pages.
- **Single-user:** The Flask app uses in-memory job storage. It's designed for local/single-user use, not production deployment.
- **First-time download:** ~140 MB of model weights are downloaded on first run.

## Acknowledgments

- **[manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)** by qweasdd — the core colorization model (vendored under `vendor/`)
- **[FFDNet](https://github.com/cszn/FFDNet)** — denoising network used for preprocessing
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — PDF extraction and reassembly

## License

This project is provided under the [MIT License](LICENSE).

The vendored manga-colorization-v2 model code and FFDNet denoiser code retain their original licenses. See the respective repositories for details.

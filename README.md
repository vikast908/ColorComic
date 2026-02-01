# ColorComic

Automatically colorize black-and-white comic and manga pages using deep learning. Upload a PDF, get back a fully colorized version — no API keys, no cloud services, everything runs locally on your machine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Two colorization modes**
  - **Auto** — fully automatic, no reference needed (manga-colorization-v2)
  - **Reference** — upload one colored page for higher quality results (MangaNinja, CVPR 2025)
- **Post-processing pipeline** — L-channel preservation for perfect line fidelity, guided filter for clean edges
- **Optional 4x upscaling** — built-in Real-ESRGAN for print-quality output
- **VRAM-aware model management** — only one colorizer loaded at a time, safe for 8 GB GPUs
- **GPU accelerated** — 2-5 seconds per page in auto mode on CUDA GPUs, with automatic CPU fallback
- **GPU detection** — analyze your hardware specs before choosing a device
- **Cross-page color consistency** — LAB color transfer keeps character/environment colors consistent across pages (auto mode)
- **Live preview** — side-by-side original vs. colorized comparison updates in real-time during processing
- **PDF in, PDF out** — upload a B&W comic PDF, download a colorized PDF
- **Zero cloud dependency** — everything runs locally, no API keys needed
- **Auto model download** — weights are downloaded automatically on first use

## How It Works

### Auto Mode (existing + post-processing)

```
Upload PDF → Extract pages at 300 DPI
  → For each page:
      mc-v2 colorize (576×576)
      → L-channel preservation (replace L with original)
      → Guided filter (clean edge bleeding)
      → [Optional] Real-ESRGAN 4x upscale
      → Color consistency (LAB transfer, pages 2+)
  → Reassemble PDF → Preview/Download
```

### Reference Mode (MangaNinja)

```
Upload PDF + reference image → Extract pages
  → For each page:
      MangaNinja colorize (512×512, using reference)
      → L-channel preservation
      → Guided filter
      → [Optional] Real-ESRGAN 4x upscale
  → Reassemble PDF → Preview/Download
```

The **auto mode** uses [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2), a U-Net with an SEResNeXt encoder trained on manga artwork. Cross-page consistency uses Reinhard LAB color transfer on chrominance channels.

The **reference mode** uses [MangaNinja](https://github.com/ali-vilab/MangaNinjia) (CVPR 2025), which takes a colored reference page and transfers its color palette to all target pages using a dual UNet architecture with reference attention and point correspondence.

## Requirements

- **Python** 3.10+
- **PyTorch** 2.3+ (with CUDA for GPU acceleration)
- **~500 MB disk space** for auto mode weights (downloaded automatically)
- **~6 GB disk space** for reference mode weights (downloaded on first use)
- **GPU (optional):** Any NVIDIA GPU with 2+ GB VRAM for auto mode, 6+ GB for reference mode

### VRAM Usage

| Mode | VRAM | Speed |
|------|------|-------|
| Auto (mc-v2) | ~3 GB | ~3 s/page |
| Auto + ESRGAN | ~3.5 GB | ~5 s/page |
| Reference (MangaNinja) | ~6 GB | ~15-30 s/page |

Only one colorizer is loaded at a time. Switching modes automatically unloads the current model and frees VRAM.

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
COLORCOMIC_DEVICE=auto                # auto | cpu | cuda
COLOR_TRANSFER_STRENGTH=0.7           # 0.0 = no transfer, 1.0 = full transfer
POSTPROCESS_L_CHANNEL=1               # 1 = enabled, 0 = disabled
POSTPROCESS_GUIDED_FILTER=1           # 1 = enabled, 0 = disabled
POSTPROCESS_UPSCALE=0                 # 1 = enable Real-ESRGAN 4x upscale
MANGANINJA_DENOISE_STEPS=30           # Denoising steps for reference mode
```

### 6. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. On first run, auto mode weights (~140 MB) are downloaded from Google Drive. Reference mode weights (~6 GB) are downloaded from HuggingFace on first use.

## Usage

1. **Upload** — Drop a B&W comic/manga PDF onto the upload page
2. **Choose mode** — Select "Auto" for automatic colorization or "Reference" for reference-based (upload a colored reference image)
3. **Detect GPU** — Click "Detect GPU" to see your hardware specs and pick CPU or GPU
4. **Colorize** — Hit "Upload & Colorize" and watch the progress with live previews
5. **Download** — Get the colorized PDF or review individual pages

## Configuration

| Variable | Default | Description |
|---|---|---|
| `COLORCOMIC_DEVICE` | `auto` | Device for inference. `auto` picks GPU if available. |
| `COLOR_TRANSFER_STRENGTH` | `0.7` | Cross-page color alignment strength (0.0–1.0). Auto mode only. |
| `POSTPROCESS_L_CHANNEL` | `1` | Replace colorized luminance with original grayscale for sharper lines. |
| `POSTPROCESS_GUIDED_FILTER` | `1` | Smooth color bleeding at edges using the original as guide. |
| `POSTPROCESS_UPSCALE` | `0` | Enable Real-ESRGAN 4x upscaling (downloads ~17 MB model on first use). |
| `MANGANINJA_DENOISE_STEPS` | `30` | DDIM denoising steps for reference mode. Lower = faster, higher = better quality. |
| `SD15_MODEL_PATH` | HuggingFace | Override Stable Diffusion 1.5 model path for reference mode. |
| `CLIP_VISION_PATH` | HuggingFace | Override CLIP vision model path for reference mode. |
| `SECRET_KEY` | random | Flask session secret. Set to a fixed string in production. |

## Project Structure

```
ColorComic/
├── app.py                    # Flask application & routes
├── config.py                 # Configuration (env vars, paths)
├── requirements.txt          # Python dependencies
│
├── core/
│   ├── model_manager.py      # VRAM-aware model switching (one colorizer at a time)
│   ├── ml_colorizer.py       # manga-colorization-v2 wrapper (auto mode)
│   ├── manga_ninja_colorizer.py  # MangaNinja wrapper (reference mode)
│   ├── postprocessor.py      # L-channel preservation + guided filter
│   ├── upscaler.py           # Real-ESRGAN 4x upscaler (self-contained)
│   ├── color_consistency.py  # LAB color transfer for cross-page consistency
│   ├── model_downloader.py   # Auto-download weights (Google Drive + HuggingFace)
│   ├── pdf_handler.py        # PDF extraction & reassembly (PyMuPDF)
│   └── panel_detector.py     # Panel detection (available for future use)
│
├── models/
│   ├── schemas.py            # Pydantic data models (JobState, PanelRegion)
│   └── weights/              # Model weights (auto-downloaded, gitignored)
│
├── vendor/
│   ├── manga_colorization_v2/  # Vendored mc-v2 inference code
│   └── manganinja/             # Vendored MangaNinja inference code (CC BY-NC 4.0)
│       ├── pipeline.py         # Main diffusion pipeline
│       ├── point_network.py    # PointNet for spatial correspondence
│       ├── annotator/          # Lineart extraction
│       └── models/             # Custom UNet, attention, transformer blocks
│
├── templates/                # Jinja2 HTML templates
│   ├── base.html             # Layout with dark theme
│   ├── index.html            # Upload page with mode selector + GPU detection
│   ├── processing.html       # Live progress with side-by-side preview
│   └── preview.html          # Page-by-page review & download
│
└── static/
    ├── css/style.css         # Dark theme stylesheet
    └── js/
        ├── app.js            # Global JS utilities
        └── upload.js         # Upload logic, mode toggle, reference upload
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Upload page |
| `POST` | `/upload` | Upload PDF (+ optional reference image and mode) |
| `POST` | `/api/colorize/<job_id>` | Start colorization pipeline |
| `GET` | `/api/colorize/<job_id>/stream` | SSE stream of progress events |
| `GET` | `/api/preview/<job_id>/<page>` | Serve a colorized page image |
| `GET` | `/pages/<job_id>/<page>` | Serve an original B&W page image |
| `GET` | `/api/download/<job_id>` | Download the colorized PDF |
| `GET` | `/api/gpu-info` | GPU detection (name, VRAM, compute capability) |
| `GET` | `/api/status` | Model health check (device, mode, CUDA status) |
| `GET` | `/processing/<job_id>` | Processing page with live preview |
| `GET` | `/preview/<job_id>` | Review colorized pages |

## GPU Support

ColorComic auto-detects CUDA GPUs at startup. On the upload page, click **"Detect GPU"** to see:

- GPU name and model
- Total and free VRAM
- Compute capability and SM count
- CUDA toolkit version
- A recommendation (GPU or CPU based on available VRAM)

**Minimum:** Any NVIDIA GPU with 2 GB VRAM (auto mode)
**Recommended:** 6+ GB VRAM for reference mode (RTX 3060, RTX 4070, etc.)
**CPU fallback:** Automatic if GPU runs out of memory mid-inference

If PyTorch was installed without CUDA support (`torch+cpu`), GPU will not be available regardless of hardware. Reinstall with the correct CUDA index URL (see [Installation](#3-install-pytorch)).

## Limitations

- **Manga-optimized:** Both models are trained on manga/anime artwork. Western comics may get lower-quality results.
- **Reference mode is CC BY-NC 4.0:** MangaNinja is licensed for non-commercial use only. A notice is displayed in the UI.
- **Reference mode needs a colored reference:** You must provide one colored page. The model transfers that page's color palette to all others.
- **First-time reference download:** ~6 GB of weights (SD 1.5, CLIP, MangaNinja) are downloaded on first reference mode use.
- **Consistency is approximate:** LAB color transfer (auto mode) aligns global color distributions, not specific character elements.
- **Single-user:** The Flask app uses in-memory job storage. Designed for local/single-user use.

## Acknowledgments

- **[manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)** by qweasdd — automatic colorization model (auto mode)
- **[MangaNinja](https://github.com/ali-vilab/MangaNinjia)** by ali-vilab — reference-based colorization model (CVPR 2025, CC BY-NC 4.0)
- **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** by xinntao — anime-optimized super-resolution
- **[FFDNet](https://github.com/cszn/FFDNet)** — denoising network used for preprocessing
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — PDF extraction and reassembly

## License

This project is provided under the [MIT License](LICENSE).

The vendored manga-colorization-v2 model code and FFDNet denoiser code retain their original licenses. The vendored MangaNinja code is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only). See the respective repositories for details.

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(24).hex())
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

    # Image processing
    PAGE_DPI = 300
    PREVIEW_DPI = 150

    # ML model
    WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights")
    GENERATOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "generator.zip")
    EXTRACTOR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "extractor.pth")
    DENOISER_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "denoiser")
    ML_DEVICE = os.environ.get("COLORCOMIC_DEVICE", "auto")
    COLOR_TRANSFER_STRENGTH = float(os.environ.get("COLOR_TRANSFER_STRENGTH", "0.7"))

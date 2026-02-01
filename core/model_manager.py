"""VRAM-aware model manager — only one colorizer loaded at a time.

Designed for 8 GB GPUs where mc-v2 (~3 GB) and MangaNinja (~6 GB)
cannot coexist simultaneously.
"""

import gc
import threading

import torch

from config import Config
from core.ml_colorizer import MangaColorizer


class ModelManager:
    """Manages exclusive loading of colorizer models to fit in VRAM.

    Usage::

        manager = ModelManager(device="auto")
        colorizer = manager.get_colorizer("auto")       # loads mc-v2
        colorizer = manager.get_colorizer("reference")   # unloads mc-v2, loads MangaNinja
    """

    def __init__(self, device: str = "auto"):
        self._lock = threading.Lock()
        self._device = device
        self._current_mode: str | None = None
        self._colorizer = None  # MangaColorizer or MangaNinjaColorizer

    @property
    def current_mode(self) -> str | None:
        return self._current_mode

    @property
    def device_name(self) -> str:
        if self._colorizer is not None and hasattr(self._colorizer, "device_name"):
            return self._colorizer.device_name
        dev = self._resolve_device()
        return str(dev)

    @property
    def cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def _resolve_device(self):
        if self._device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device)

    def get_colorizer(self, mode: str = "auto"):
        """Return the colorizer for *mode*, loading it if necessary.

        If a different mode is currently loaded, it will be unloaded first
        to free VRAM before loading the requested one.

        Parameters
        ----------
        mode : str
            ``"auto"`` for manga-colorization-v2, ``"reference"`` for MangaNinja.
        """
        with self._lock:
            if self._current_mode == mode and self._colorizer is not None:
                return self._colorizer

            # Unload current model
            self._unload()

            if mode == "auto":
                self._colorizer = self._load_mcv2()
            elif mode == "reference":
                self._colorizer = self._load_manganinja()
            else:
                raise ValueError(f"Unknown colorization mode: {mode!r}")

            self._current_mode = mode
            return self._colorizer

    def switch_device(self, device: str):
        """Change the target device. Reloads the current model if loaded."""
        with self._lock:
            if device == self._device:
                return
            self._device = device
            if self._colorizer is not None:
                mode = self._current_mode
                self._unload()
                # Re-acquire lock is not needed since we're already holding it
                if mode == "auto":
                    self._colorizer = self._load_mcv2()
                elif mode == "reference":
                    self._colorizer = self._load_manganinja()
                self._current_mode = mode

    def _unload(self):
        """Unload current model and flush VRAM."""
        if self._colorizer is not None:
            if hasattr(self._colorizer, "unload"):
                self._colorizer.unload()
            del self._colorizer
            self._colorizer = None
            self._current_mode = None
        self._flush_vram()

    @staticmethod
    def _flush_vram():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_mcv2(self) -> MangaColorizer:
        """Load the manga-colorization-v2 model."""
        from core.model_downloader import ensure_models_downloaded

        ensure_models_downloaded(Config.WEIGHTS_DIR, callback=print)
        colorizer = MangaColorizer(
            device=self._device,
            generator_path=Config.GENERATOR_WEIGHTS_PATH,
            extractor_path=Config.EXTRACTOR_WEIGHTS_PATH,
            denoiser_weights_dir=Config.DENOISER_WEIGHTS_DIR,
        )
        print(f"[ModelManager] mc-v2 loaded on {colorizer.device_name}")
        return colorizer

    def _load_manganinja(self):
        """Load the MangaNinja reference-based colorizer."""
        from core.model_downloader import ensure_manganinja_downloaded
        from core.manga_ninja_colorizer import MangaNinjaColorizer

        ensure_manganinja_downloaded(Config, callback=print)
        colorizer = MangaNinjaColorizer(
            device=self._device,
            config=Config,
        )
        print(f"[ModelManager] MangaNinja loaded on {colorizer.device_name}")
        return colorizer

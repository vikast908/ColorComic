"""ML-based manga/comic colorization using manga-colorization-v2."""

import gc
import sys
import os
import threading

import cv2
import numpy as np
import torch

# Make vendor package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vendor.manga_colorization_v2.colorizator import MangaColorizator


class MangaColorizer:
    """Singleton-style wrapper around the manga-colorization-v2 model.

    Usage::

        colorizer = MangaColorizer(device="auto",
                                   generator_path="...",
                                   extractor_path="...",
                                   denoiser_weights_dir="...")
        rgb_result = colorizer.colorize(bgr_image)
    """

    def __init__(self, device: str = "auto",
                 generator_path: str = "",
                 extractor_path: str = "",
                 denoiser_weights_dir: str = ""):
        self._lock = threading.Lock()
        self._device = self._resolve_device(device)
        self._generator_path = generator_path
        self._extractor_path = extractor_path
        self._denoiser_weights_dir = denoiser_weights_dir
        self._model = MangaColorizator(
            device=self._device,
            generator_path=generator_path,
            extractor_path=extractor_path,
            denoiser_weights_dir=denoiser_weights_dir,
        )
        self.device_name = str(self._device)
        self.cuda_available = torch.cuda.is_available()

    @staticmethod
    def _resolve_device(device: str):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def switch_device(self, device: str) -> None:
        """Reload the model on a different device if needed."""
        target = self._resolve_device(device)
        if str(target) == self.device_name:
            return
        with self._lock:
            self._device = target
            self._model = MangaColorizator(
                device=self._device,
                generator_path=self._generator_path,
                extractor_path=self._extractor_path,
                denoiser_weights_dir=self._denoiser_weights_dir,
            )
            self.device_name = str(self._device)

    def colorize(self, image: np.ndarray, size: int = 576) -> np.ndarray:
        """Colorize a single B&W page image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (as read by ``cv2.imread``).
        size : int
            Target resize dimension (must be divisible by 32).

        Returns
        -------
        np.ndarray
            Colorized image in BGR uint8 format, same spatial dimensions
            as the input.
        """
        # Convert BGR -> RGB for the model
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        with self._lock:
            try:
                with torch.inference_mode():
                    self._model.set_image(rgb, size=size, apply_denoise=True)
                    result = self._model.colorize()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self.device_name != "cpu":
                    # OOM fallback: retry on CPU
                    torch.cuda.empty_cache()
                    self._device = torch.device("cpu")
                    self._model = MangaColorizator(
                        device=self._device,
                        generator_path=self._generator_path,
                        extractor_path=self._extractor_path,
                        denoiser_weights_dir=self._denoiser_weights_dir,
                    )
                    self.device_name = "cpu"
                    with torch.inference_mode():
                        self._model.set_image(rgb, size=size, apply_denoise=True)
                        result = self._model.colorize()
                else:
                    raise

        # result is float32 RGB in [0, 1] — convert to BGR uint8
        result_uint8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)

        # Resize back to original dimensions
        if result_bgr.shape[:2] != (orig_h, orig_w):
            # INTER_AREA for downsampling, INTER_LANCZOS4 for upsampling
            rh, rw = result_bgr.shape[:2]
            interp = cv2.INTER_AREA if (rh > orig_h or rw > orig_w) else cv2.INTER_LANCZOS4
            result_bgr = cv2.resize(result_bgr, (orig_w, orig_h), interpolation=interp)

        return result_bgr

    def unload(self):
        """Release model and free GPU memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

"""MangaNinja reference-based colorizer wrapper.

Loads the MangaNinja pipeline (SD 1.5 + Reference UNet + Denoising UNet +
ControlNet + CLIP + PointNet) and provides a ``colorize()`` interface
matching :class:`MangaColorizer`.

Licensed under CC BY-NC 4.0 — non-commercial use only.
"""

import gc
import threading

import cv2
import numpy as np
import PIL.Image
import torch


class MangaNinjaColorizer:
    """Reference-based manga colorization using MangaNinja (CVPR 2025).

    Usage::

        colorizer = MangaNinjaColorizer(device="auto", config=Config)
        result_bgr = colorizer.colorize(bgr_image, reference_image=ref_bgr)
    """

    def __init__(self, device: str = "auto", config=None):
        self._lock = threading.Lock()
        self._device = self._resolve_device(device)
        self._config = config
        self._pipeline = None
        self.device_name = str(self._device)
        self.cuda_available = torch.cuda.is_available()

        self._load_pipeline()

    @staticmethod
    def _resolve_device(device: str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_pipeline(self):
        """Load the full MangaNinja pipeline."""
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            DDIMScheduler,
        )
        from transformers import (
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTokenizer,
            CLIPVisionModelWithProjection,
        )
        from vendor.manganinja.pipeline import MangaNinjiaPipeline
        from vendor.manganinja.models.unet_2d_condition import UNet2DConditionModel
        from vendor.manganinja.models.refunet_2d_condition import RefUNet2DConditionModel
        from vendor.manganinja.point_network import PointNet
        from vendor.manganinja.annotator.lineart import BatchLineartDetector

        cfg = self._config
        device = self._device
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        print("[MangaNinja] Loading SD 1.5 components...")

        # Scheduler
        scheduler = DDIMScheduler.from_pretrained(cfg.SD15_MODEL_PATH, subfolder="scheduler")

        # VAE
        vae = AutoencoderKL.from_pretrained(cfg.SD15_MODEL_PATH, subfolder="vae")

        # Denoising UNet
        denoising_unet = UNet2DConditionModel.from_pretrained(
            cfg.SD15_MODEL_PATH, subfolder="unet",
            in_channels=4, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
        )
        print("[MangaNinja] Loading denoising UNet weights...")
        state = torch.load(cfg.MANGANINJA_DENOISING_UNET, map_location="cpu")
        denoising_unet.load_state_dict(state, strict=False)
        del state

        # Reference UNet
        reference_unet = RefUNet2DConditionModel.from_pretrained(
            cfg.SD15_MODEL_PATH, subfolder="unet",
            in_channels=4, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
        )
        print("[MangaNinja] Loading reference UNet weights...")
        state = torch.load(cfg.MANGANINJA_REFERENCE_UNET, map_location="cpu")
        reference_unet.load_state_dict(state, strict=False)
        del state

        # ControlNet
        controlnet = ControlNetModel.from_pretrained(
            cfg.CONTROLNET_LINEART_PATH,
            in_channels=4, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
        )
        print("[MangaNinja] Loading ControlNet weights...")
        state = torch.load(cfg.MANGANINJA_CONTROLNET, map_location="cpu")
        controlnet.load_state_dict(state, strict=False)
        del state

        # CLIP
        print("[MangaNinja] Loading CLIP...")
        tokenizer = CLIPTokenizer.from_pretrained(cfg.CLIP_VISION_PATH)
        text_encoder = CLIPTextModel.from_pretrained(cfg.CLIP_VISION_PATH)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.CLIP_VISION_PATH)

        # PointNet
        point_net = PointNet()
        state = torch.load(cfg.MANGANINJA_POINTNET, map_location="cpu")
        point_net.load_state_dict(state, strict=False)
        del state

        # Lineart preprocessor
        preprocessor = BatchLineartDetector(cfg.LINEART_ANNOTATOR_PATH)
        preprocessor.to(device, dtype=torch.float32)

        # Build pipeline
        self._pipeline = MangaNinjiaPipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            controlnet=controlnet,
            scheduler=scheduler,
            refnet_tokenizer=tokenizer,
            refnet_text_encoder=text_encoder,
            refnet_image_encoder=image_encoder,
            controlnet_tokenizer=tokenizer,
            controlnet_text_encoder=text_encoder,
            controlnet_image_encoder=image_encoder,
            point_net=point_net,
            preprocessor=preprocessor,
        )

        self._pipeline = self._pipeline.to(device=device, dtype=dtype)
        print(f"[MangaNinja] Pipeline loaded on {device}")

    def colorize(self, image: np.ndarray, reference_image: np.ndarray = None,
                 size: int = 512) -> np.ndarray:
        """Colorize a single B&W page using a colored reference.

        Parameters
        ----------
        image : np.ndarray
            Grayscale input page in BGR uint8.
        reference_image : np.ndarray
            Colored reference page in BGR uint8.
        size : int
            Processing resolution (512 recommended).

        Returns
        -------
        np.ndarray
            Colorized image in BGR uint8, same dimensions as input.
        """
        if reference_image is None:
            raise ValueError("MangaNinja requires a reference_image")

        orig_h, orig_w = image.shape[:2]

        # Convert BGR -> RGB PIL
        target_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

        target_pil = PIL.Image.fromarray(target_rgb)
        ref_pil = PIL.Image.fromarray(ref_rgb)

        with self._lock:
            result_rgb = self._pipeline(
                ref_image=ref_pil,
                target_image=target_pil,
                num_inference_steps=self._config.MANGANINJA_DENOISE_STEPS,
                width=size,
                height=size,
            )

        # Convert RGB -> BGR
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        # Resize back to original dimensions
        if result_bgr.shape[:2] != (orig_h, orig_w):
            result_bgr = cv2.resize(result_bgr, (orig_w, orig_h),
                                    interpolation=cv2.INTER_LANCZOS4)

        return result_bgr

    def unload(self):
        """Release pipeline and free GPU memory."""
        with self._lock:
            if self._pipeline is not None:
                del self._pipeline
                self._pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

"""MangaNinja inference pipeline (from inference/manganinjia_pipeline.py).

DiffusionPipeline subclass that performs reference-based manga colorization
using a reference UNet, denoising UNet, ControlNet, CLIP, and PointNet.
"""

from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DiffusionPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from .annotator.lineart import BatchLineartDetector
from .image_util import resize_max_res
from .models.mutual_self_attention_multi_scale import ReferenceAttentionControl
from .models.refunet_2d_condition import RefUNet2DConditionModel
from .models.unet_2d_condition import UNet2DConditionModel
from .point_network import PointNet


class MangaNinjiaPipeline(DiffusionPipeline):
    """Reference-based manga colorization pipeline.

    Implements the MangaNinja (CVPR 2025) approach: given a grayscale manga
    page and one colored reference page, transfers the reference's colors
    to the target using dual classifier-free guidance with point correspondence.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        reference_unet: RefUNet2DConditionModel,
        denoising_unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: DDIMScheduler,
        refnet_tokenizer: CLIPTokenizer,
        refnet_text_encoder: CLIPTextModel,
        refnet_image_encoder: CLIPVisionModelWithProjection,
        controlnet_tokenizer: CLIPTokenizer,
        controlnet_text_encoder: CLIPTextModel,
        controlnet_image_encoder: CLIPVisionModelWithProjection,
        point_net: PointNet,
        preprocessor: BatchLineartDetector,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            controlnet=controlnet,
            scheduler=scheduler,
            refnet_tokenizer=refnet_tokenizer,
            refnet_text_encoder=refnet_text_encoder,
            refnet_image_encoder=refnet_image_encoder,
            controlnet_tokenizer=controlnet_tokenizer,
            controlnet_text_encoder=controlnet_text_encoder,
            controlnet_image_encoder=controlnet_image_encoder,
            point_net=point_net,
            preprocessor=preprocessor,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.ref_image_processor = CLIPImageProcessor()
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )

    @torch.no_grad()
    def __call__(
        self,
        ref_image: PIL.Image.Image,
        target_image: PIL.Image.Image,
        point_map_ref: Optional[torch.Tensor] = None,
        point_map_target: Optional[torch.Tensor] = None,
        num_inference_steps: int = 30,
        guidance_scale_ref: float = 2.5,
        guidance_scale_point: float = 2.5,
        width: int = 512,
        height: int = 512,
        generator: Optional[torch.Generator] = None,
    ) -> np.ndarray:
        """Run colorization.

        Parameters
        ----------
        ref_image : PIL.Image.Image
            Colored reference image (RGB).
        target_image : PIL.Image.Image
            Grayscale target image to colorize (RGB or L mode).
        point_map_ref : torch.Tensor, optional
            Point correspondence map for reference, ``(1, 1, H, W)``.
        point_map_target : torch.Tensor, optional
            Point correspondence map for target, ``(1, 1, H, W)``.
        num_inference_steps : int
            Number of DDIM denoising steps.
        guidance_scale_ref : float
            Classifier-free guidance scale for reference features.
        guidance_scale_point : float
            Classifier-free guidance scale for point features.
        width, height : int
            Processing resolution (should be 512).

        Returns
        -------
        np.ndarray
            Colorized image as uint8 RGB, shape ``(H, W, 3)``.
        """
        device = self._execution_device
        dtype = self.denoising_unet.dtype

        # 1. Prepare images
        ref_pil = ref_image.convert("RGB").resize((width, height), PIL.Image.LANCZOS)
        target_pil = target_image.convert("RGB").resize((width, height), PIL.Image.LANCZOS)

        # Reference image for CLIP
        clip_ref = self.ref_image_processor(images=ref_pil, return_tensors="pt").pixel_values
        clip_ref = clip_ref.to(device=device, dtype=dtype)

        # Reference image for VAE
        ref_tensor = torch.from_numpy(np.array(ref_pil)).float() / 127.5 - 1.0
        ref_tensor = ref_tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

        # Target image for lineart
        target_tensor = torch.from_numpy(np.array(target_pil)).float() / 255.0
        target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

        # 2. Extract line art from target
        lineart = self.preprocessor(target_tensor)  # (1, 1, H, W)
        lineart_3ch = lineart.repeat(1, 3, 1, 1)  # ControlNet expects 3 channels

        # 3. CLIP encode reference
        clip_hidden = self.refnet_image_encoder(clip_ref).image_embeds
        clip_hidden = clip_hidden.unsqueeze(1)  # (1, 1, D)

        # 4. Text encode (empty prompt for unconditional)
        uncond_tokens = self.refnet_tokenizer(
            [""], padding="max_length",
            max_length=self.refnet_tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        uncond_embeds = self.refnet_text_encoder(uncond_tokens)[0]  # (1, seq, D)

        # 5. Point embeddings
        point_emb_ref = None
        point_emb_main = None
        if point_map_ref is not None and point_map_target is not None:
            point_map_ref = point_map_ref.to(device=device, dtype=dtype)
            point_map_target = point_map_target.to(device=device, dtype=dtype)
            point_emb_ref = self.point_net(point_map_ref)
            point_emb_main = self.point_net(point_map_target)

        # 6. Encode reference to latent
        ref_latent = self.vae.encode(ref_tensor).latent_dist.sample() * self.vae.config.scaling_factor

        # 7. Set up reference attention
        writer_ctrl = ReferenceAttentionControl(self.reference_unet, mode="write")
        reader_ctrl = ReferenceAttentionControl(self.denoising_unet, mode="read")
        writer_ctrl.register()
        reader_ctrl.register()

        # 8. Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 9. Start with noise
        latent_shape = (1, self.denoising_unet.config.in_channels,
                        height // self.vae_scale_factor,
                        width // self.vae_scale_factor)
        latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
        latents = latents * self.scheduler.init_noise_sigma

        # 10. Encode lineart for ControlNet
        lineart_latent = self.vae.encode(
            lineart_3ch * 2.0 - 1.0  # normalize to [-1, 1]
        ).latent_dist.sample() * self.vae.config.scaling_factor

        # 11. Build encoder_hidden_states for CFG
        # 3-way: uncond, ref, ref+point
        encoder_hidden_states = torch.cat([uncond_embeds, clip_hidden, clip_hidden], dim=0)

        # 12. Denoising loop
        for i, t in enumerate(timesteps):
            # Reference UNet pass (only at step 0)
            if i == 0:
                writer_ctrl.clear()
                self.reference_unet(
                    ref_latent,
                    t,
                    encoder_hidden_states=clip_hidden,
                )
                reader_ctrl.update(
                    writer_ctrl,
                    point_embeddings_ref=point_emb_ref,
                    point_embeddings_main=point_emb_main,
                )

            # Expand latents for 3-way CFG
            latent_model_input = torch.cat([latents] * 3, dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # ControlNet
            control_input = torch.cat([lineart_latent] * 3, dim=0)
            down_block_res, mid_block_res = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control_input,
                return_dict=False,
            )

            # Denoising UNet
            noise_pred = self.denoising_unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res,
                mid_block_additional_residual=mid_block_res,
                return_dict=False,
            )[0]

            # 3-way CFG: uncond, ref, point
            noise_uncond, noise_ref, noise_point = noise_pred.chunk(3)

            # Dual guidance
            noise_1 = noise_uncond + guidance_scale_ref * (noise_ref - noise_uncond)
            noise_2 = noise_ref + guidance_scale_point * (noise_point - noise_ref)
            noise_pred_combined = (noise_1 + noise_2) / 2.0

            # DDIM step
            latents = self.scheduler.step(noise_pred_combined, t, latents).prev_sample

        # 13. Decode
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample

        # 14. Post-process to numpy
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).astype(np.uint8)

        # Cleanup
        writer_ctrl.unregister()
        reader_ctrl.unregister()
        writer_ctrl.clear()
        reader_ctrl.clear()

        return image

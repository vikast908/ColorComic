import os

import torch
from torchvision.transforms import ToTensor
import numpy as np
from .networks.models import Colorizer
from .denoising.denoiser import FFDNetDenoiser
from .utils.utils import resize_pad


class MangaColorizator:
    def __init__(self, device, generator_path='networks/generator.zip',
                 extractor_path='networks/extractor.pth',
                 denoiser_weights_dir='denoising/models/'):
        self.device = device
        self.colorizer = Colorizer().to(device)

        generator_state = torch.load(generator_path, map_location=device,
                                     weights_only=False)
        self.colorizer.generator.load_state_dict(generator_state)

        # Load separate extractor weights only if available.
        # When the generator state dict already contains encoder.* keys
        # (which it does for the standard release), this is not needed.
        if extractor_path and os.path.isfile(extractor_path):
            extractor_state = torch.load(extractor_path, map_location=device,
                                         weights_only=False)
            self.colorizer.generator.encoder.load_state_dict(extractor_state)

        self.colorizer.eval()

        self.denoiser = FFDNetDenoiser(
            _device=device if isinstance(device, str) else str(device),
            _weights_dir=denoiser_weights_dir
        )

        self.current_image = None
        self.current_hint = None
        self.current_pad = None

    def set_image(self, image, size=576, apply_denoise=True, denoise_sigma=25,
                  transform=ToTensor()):
        if size % 32 != 0:
            raise RuntimeError("size is not divisible by 32")

        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma=denoise_sigma)

        image, self.current_pad = resize_pad(image, size)

        self.current_image = transform(image).unsqueeze(0).to(self.device)
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2],
                                        self.current_image.shape[3]).to(self.device)

    def update_hint(self, hint, mask):
        if isinstance(hint, np.ndarray):
            hint = hint.astype('float32')
            if hint.max() > 1.0:
                hint = hint / 255.0
            hint = (hint - 0.5) / 0.5
            hint = torch.FloatTensor(hint).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if isinstance(mask, np.ndarray):
            mask = mask.astype('float32')
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.device)

        self.current_hint = torch.cat([hint, mask], 1)

    @torch.no_grad()
    def colorize(self):
        fake_color, _ = self.colorizer(torch.cat([self.current_image, self.current_hint], 1))
        result = fake_color[0].detach().cpu().permute(1, 2, 0).numpy()
        result = result * 0.5 + 0.5

        if self.current_pad[0] > 0:
            result = result[:-self.current_pad[0]]
        if self.current_pad[1] > 0:
            result = result[:, :-self.current_pad[1]]

        return result

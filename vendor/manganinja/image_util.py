"""Image utility functions (from MangaNinja utils/image_util.py)."""

import torch


def resize_max_res(img: torch.Tensor, max_edge_resolution: int) -> torch.Tensor:
    """Resize image so the longer edge equals *max_edge_resolution*.

    Parameters
    ----------
    img : torch.Tensor
        Image tensor of shape ``(B, C, H, W)``.
    max_edge_resolution : int
        Target size for the longer edge.

    Returns
    -------
    torch.Tensor
        Resized image tensor.
    """
    _, _, h, w = img.shape
    scale = max_edge_resolution / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    # Ensure divisible by 8
    new_h = new_h - (new_h % 8)
    new_w = new_w - (new_w % 8)
    return torch.nn.functional.interpolate(
        img, size=(new_h, new_w), mode="bilinear", align_corners=False
    )


def chw2hwc(chw: torch.Tensor) -> torch.Tensor:
    """Convert ``(C, H, W)`` tensor to ``(H, W, C)``."""
    return chw.permute(1, 2, 0)

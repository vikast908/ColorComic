"""Post-processing pipeline for colorized manga pages.

Steps (each toggleable):
1. L-channel preservation — replace colorized L with original grayscale L in LAB.
2. Guided filter — smooth color bleeding at edges using original as guide.
3. Real-ESRGAN 4x upscale (optional).
"""

import cv2
import numpy as np

# Max edge length for guided filter processing.  Larger images are
# downscaled before filtering to avoid multi-second CPU stalls on
# high-DPI pages (e.g. 300 DPI → ~2500×3500 px).
_GUIDED_FILTER_MAX_EDGE = 1024


class PostProcessor:
    """Applies post-processing to improve colorized output quality."""

    def __init__(self, *, l_channel: bool = True, guided_filter: bool = True,
                 upscale: bool = False, upscaler=None):
        self.l_channel = l_channel
        self.guided_filter = guided_filter
        self.upscale = upscale
        self._upscaler = upscaler

    def process(self, colorized: np.ndarray, original_gray: np.ndarray) -> np.ndarray:
        """Run the post-processing pipeline.

        Parameters
        ----------
        colorized : np.ndarray
            Colorized image in BGR uint8, same size as *original_gray*.
        original_gray : np.ndarray
            Original B&W page in BGR uint8 (may be single-channel or 3-channel grayscale).

        Returns
        -------
        np.ndarray
            Processed BGR uint8 image.
        """
        result = colorized

        if self.l_channel:
            result = self._preserve_l_channel(result, original_gray)

        if self.guided_filter:
            result = self._apply_guided_filter(result, original_gray)

        if self.upscale and self._upscaler is not None:
            result = self._upscaler.upscale(result)

        return result

    @staticmethod
    def _preserve_l_channel(colorized: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Replace the L channel of *colorized* with the L of *original*.

        This guarantees that line work from the original is perfectly preserved
        while keeping the chrominance (color) from the colorization model.
        """
        if len(original.shape) == 3 and original.shape[2] == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original

        ch, cw = colorized.shape[:2]
        if gray.shape[:2] != (ch, cw):
            gray = cv2.resize(gray, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = gray
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_guided_filter(colorized: np.ndarray, original: np.ndarray,
                             radius: int = 4, eps: float = 0.02) -> np.ndarray:
        """Apply guided filter using *original* as guide to clean color bleeding.

        Large images are downscaled to *_GUIDED_FILTER_MAX_EDGE* before
        filtering, then the filtered chrominance is upscaled back.  This
        keeps per-page time well under 1 s even at 300 DPI.
        """
        if len(original.shape) == 3 and original.shape[2] == 3:
            guide = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            guide = original

        full_h, full_w = colorized.shape[:2]
        if guide.shape[:2] != (full_h, full_w):
            guide = cv2.resize(guide, (full_w, full_h), interpolation=cv2.INTER_LANCZOS4)

        # Decide whether to downscale for performance
        max_edge = max(full_h, full_w)
        need_downscale = max_edge > _GUIDED_FILTER_MAX_EDGE

        if need_downscale:
            scale = _GUIDED_FILTER_MAX_EDGE / max_edge
            small_w = int(full_w * scale)
            small_h = int(full_h * scale)
            work_color = cv2.resize(colorized, (small_w, small_h), interpolation=cv2.INTER_AREA)
            work_guide = cv2.resize(guide, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            work_color = colorized
            work_guide = guide

        # Work in LAB — filter only chrominance channels
        lab = cv2.cvtColor(work_color, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        guide_f = work_guide.astype(np.float32) / 255.0

        lab[:, :, 1] = cv2.ximgproc.guidedFilter(guide_f, lab[:, :, 1], radius, eps)
        lab[:, :, 2] = cv2.ximgproc.guidedFilter(guide_f, lab[:, :, 2], radius, eps)

        if need_downscale:
            # Extract filtered A/B at small size, upscale back to full
            a_filtered = np.clip(lab[:, :, 1] * 255.0, 0, 255).astype(np.uint8)
            b_filtered = np.clip(lab[:, :, 2] * 255.0, 0, 255).astype(np.uint8)
            a_full = cv2.resize(a_filtered, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
            b_full = cv2.resize(b_filtered, (full_w, full_h), interpolation=cv2.INTER_LINEAR)

            # Rebuild full-res LAB with original L + filtered A,B
            full_lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
            full_lab[:, :, 1] = a_full
            full_lab[:, :, 2] = b_full
            return cv2.cvtColor(full_lab, cv2.COLOR_LAB2BGR)

        lab = np.clip(lab * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

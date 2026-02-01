"""Cross-page color consistency via LAB color transfer."""

import cv2
import numpy as np


class ColorConsistencyManager:
    """Transfers chrominance from a reference page to subsequent pages.

    Uses the Reinhard LAB color transfer method, restricted to the A and B
    channels only (preserving the model's luminance decisions).  Ink-line and
    white-area pixels are masked out so they don't skew the statistics.
    """

    def __init__(self):
        self._ref_mean_a: float | None = None
        self._ref_std_a: float | None = None
        self._ref_mean_b: float | None = None
        self._ref_std_b: float | None = None

    @property
    def has_reference(self) -> bool:
        return self._ref_mean_a is not None

    @staticmethod
    def _color_mask(bgr: np.ndarray) -> np.ndarray:
        """Return a boolean mask of pixels that are neither ink nor white."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return (gray > 30) & (gray < 240)

    def set_reference(self, image: np.ndarray) -> None:
        """Store LAB chrominance statistics from the reference page.

        Parameters
        ----------
        image : np.ndarray
            Colorized page in BGR uint8 format (the first page).
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = self._color_mask(image)

        if mask.sum() < 100:
            # Not enough colored pixels — skip
            return

        a_vals = lab[mask, 1]
        b_vals = lab[mask, 2]

        self._ref_mean_a = float(np.mean(a_vals))
        self._ref_std_a = float(np.std(a_vals)) or 1.0
        self._ref_mean_b = float(np.mean(b_vals))
        self._ref_std_b = float(np.std(b_vals)) or 1.0

    def apply(self, image: np.ndarray, strength: float = 0.7) -> np.ndarray:
        """Transfer chrominance from the reference to *image*.

        Parameters
        ----------
        image : np.ndarray
            Colorized page in BGR uint8 format.
        strength : float
            Blending strength in [0, 1]. 0 = no transfer, 1 = full transfer.

        Returns
        -------
        np.ndarray
            Color-transferred image in BGR uint8 format.
        """
        if not self.has_reference:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = self._color_mask(image)

        if mask.sum() < 100:
            return image

        # Current page statistics (masked) — direct indexing avoids intermediate views
        a_vals = lab[mask, 1]
        b_vals = lab[mask, 2]

        src_mean_a = float(np.mean(a_vals))
        src_std_a = float(np.std(a_vals)) or 1.0
        src_mean_b = float(np.mean(b_vals))
        src_std_b = float(np.std(b_vals)) or 1.0

        # Reinhard transfer — in-place to avoid copies
        # A channel: out = orig + strength * (transferred - orig)
        #          = orig * (1 - strength) + transferred * strength
        a_scale = self._ref_std_a / src_std_a
        a_shift = self._ref_mean_a - src_mean_a * a_scale
        lab[:, :, 1] += strength * (lab[:, :, 1] * (a_scale - 1.0) + a_shift)

        # B channel
        b_scale = self._ref_std_b / src_std_b
        b_shift = self._ref_mean_b - src_mean_b * b_scale
        lab[:, :, 2] += strength * (lab[:, :, 2] * (b_scale - 1.0) + b_shift)

        # Clamp and convert back in one step
        np.clip(lab[:, :, 1], 0, 255, out=lab[:, :, 1])
        np.clip(lab[:, :, 2], 0, 255, out=lab[:, :, 2])

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

"""Download manga-colorization-v2 model weights if not present."""

import os

import gdown


# Google Drive file IDs for the model weights
_GENERATOR_ID = "1qmxUEKADkEM4iYLp1fpPLLKnfZ6tcF-t"
_DENOISER_ID = "161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3"


def _gdrive_download(file_id: str, dest: str, label: str, callback=None):
    """Download a file from Google Drive by *file_id* to *dest*."""
    if os.path.exists(dest):
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if callback:
        callback(f"Downloading {label}...")

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest, quiet=False)

    if callback:
        callback(f"Downloaded {label}")


def ensure_models_downloaded(weights_dir: str, callback=None):
    """Download all required model weights into *weights_dir*.

    Directory layout after download::

        weights_dir/
            generator.zip          (main colorizer generator — also used by torch.load)
            extractor.pth          (SEResNeXt feature extractor — extracted from generator.zip if bundled)
            denoiser/
                net_rgb.pth        (FFDNet denoiser)

    Parameters
    ----------
    weights_dir : str
        Path to the directory where weights will be stored.
    callback : callable, optional
        ``callback(message: str)`` called with status strings.
    """
    os.makedirs(weights_dir, exist_ok=True)

    generator_path = os.path.join(weights_dir, "generator.zip")
    extractor_path = os.path.join(weights_dir, "extractor.pth")
    denoiser_dir = os.path.join(weights_dir, "denoiser")
    denoiser_path = os.path.join(denoiser_dir, "net_rgb.pth")

    _gdrive_download(_GENERATOR_ID, generator_path, "generator weights (~400 MB)", callback)

    # The extractor weights may be bundled inside generator.zip.
    # If not available as a separate file, try to extract from the zip.
    if not os.path.exists(extractor_path):
        import zipfile

        if zipfile.is_zipfile(generator_path):
            if callback:
                callback("Extracting extractor weights from generator.zip...")
            with zipfile.ZipFile(generator_path, "r") as zf:
                names = zf.namelist()
                # Look for extractor.pth inside the zip
                extractor_names = [n for n in names if "extractor" in n.lower()]
                if extractor_names:
                    with zf.open(extractor_names[0]) as src, open(extractor_path, "wb") as dst:
                        dst.write(src.read())
                    if callback:
                        callback("Extracted extractor weights")

    # If still missing, the generator.zip is actually a torch state_dict
    # (PyTorch uses .zip extension for its save format). In this case,
    # the extractor weights may need to be downloaded separately or the
    # generator state_dict contains the encoder weights already.
    # We'll let the model loading handle it — the SEResNeXt encoder is
    # initialized inside the Generator and its weights are part of
    # generator.zip's state_dict.

    _gdrive_download(_DENOISER_ID, denoiser_path, "denoiser weights (~7 MB)", callback)

    return generator_path, extractor_path, denoiser_dir

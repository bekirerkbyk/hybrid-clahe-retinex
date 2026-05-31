"""
Image quality metrics.

Full-reference (require a clean ground-truth):
    - PSNR (Peak Signal-to-Noise Ratio)         : higher is better
    - SSIM (Structural Similarity Index)         : higher is better

No-reference (do not require ground-truth):
    - BRISQUE (lower is better)
    - Discrete entropy (higher => more information, used as a robust fallback)

BRISQUE is provided through the optional `image-quality` package. If it is not
available the function returns NaN so the rest of the pipeline still runs.

Author: Bekir Erakbiyik (2211051073)
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    import imquality.brisque as _brisque
    _HAVE_BRISQUE = True
except Exception:  # pragma: no cover
    _HAVE_BRISQUE = False


def psnr(reference_bgr, test_bgr):
    return float(sk_psnr(reference_bgr, test_bgr, data_range=255))


def ssim(reference_bgr, test_bgr):
    return float(sk_ssim(reference_bgr, test_bgr, channel_axis=2,
                         data_range=255))


def entropy(img_bgr):
    """Discrete Shannon entropy of the grayscale image (bits)."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def brisque(img_bgr):
    """No-reference BRISQUE score (lower = better perceptual quality)."""
    if not _HAVE_BRISQUE:
        return float("nan")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        return float(_brisque.score(rgb))
    except Exception:
        return float("nan")

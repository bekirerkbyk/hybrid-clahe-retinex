"""
Synthetic low-light degradation.

Because full-reference metrics (PSNR, SSIM) require a clean ground-truth
reference, we follow the common synthetic-degradation protocol: a well-exposed
reference image is darkened and corrupted with sensor noise to emulate a
night-time / low-illumination capture. The clean image is then used as the
ground truth that the enhancement algorithms try to recover.

Degradation model (operating in normalized [0,1] intensity):

    I_low = clip( scale * (I_clean ^ gamma_dark) , 0, 1 )            (under-exposure)
    I_low = I_low + Poisson_shot_noise + Gaussian_read_noise         (sensor noise)

gamma_dark > 1 compresses bright values toward zero (darkening), and a small
brightness scale further lowers the mean illumination. Shot noise scales with
intensity (signal dependent) and read noise is additive, mimicking real CMOS
sensors in which dark regions exhibit the lowest SNR.

Author: Bekir Erakbiyik (2211051073)
"""

import cv2
import numpy as np


def degrade_lowlight(img_bgr, gamma_dark=2.2, scale=0.60,
                     read_sigma=0.010, shot_peak=400.0, seed=0):
    """Create a synthetic low-light, noisy version of a clean BGR uint8 image.

    Parameters
    ----------
    gamma_dark : exponent (>1) that darkens the image.
    scale      : multiplicative brightness reduction in [0,1].
    read_sigma : std of additive Gaussian read noise (in [0,1] units).
    shot_peak  : photon scaling for the signal-dependent Poisson shot noise;
                 lower => stronger relative noise.
    seed       : RNG seed for reproducibility.

    Returns
    -------
    uint8 BGR low-light image (same shape as input).
    """
    rng = np.random.default_rng(seed)
    x = img_bgr.astype(np.float64) / 255.0

    # under-exposure
    x = scale * np.power(x, gamma_dark)

    # signal-dependent shot noise (Poisson) + additive read noise (Gaussian)
    x_shot = rng.poisson(np.clip(x, 0, 1) * shot_peak) / shot_peak
    x_noisy = x_shot + rng.normal(0.0, read_sigma, size=x.shape)

    x_noisy = np.clip(x_noisy, 0.0, 1.0)
    return (x_noisy * 255.0).astype(np.uint8)

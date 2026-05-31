"""
Enhancement methods for low-light image enhancement.

All luminance-domain methods operate on the Value (V) channel of the HSV color
space so that the original hue (H) and saturation (S) are preserved and color
distortion is minimized.

Methods implemented
-------------------
- global_histogram_equalization : GHE baseline
- gamma_correction             : fixed-gamma baseline
- clahe_v                      : CLAHE on the V channel (single component / ablation)
- single_scale_retinex_v       : SSR on the V channel (single component / ablation)
- hybrid_clahe_retinex         : the proposed CLAHE-Retinex fusion pipeline

Author: Bekir Erakbiyik (2211051073)
"""

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_hsv_v(img_bgr):
    """Return (hsv, V_uint8) for a BGR uint8 image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv, hsv[:, :, 2]


def _merge_v(hsv, v_new_uint8):
    """Replace V channel and convert HSV back to BGR uint8."""
    out = hsv.copy()
    out[:, :, 2] = np.clip(v_new_uint8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)


def _minmax_to_uint8(x):
    """Min-max normalize a float array to the 0-255 uint8 range."""
    x = x.astype(np.float64)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)  # robust min/max
    if hi - lo < 1e-6:
        hi, lo = x.max(), x.min()
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def global_histogram_equalization(img_bgr):
    """Global Histogram Equalization (GHE) on the V channel."""
    hsv, v = _to_hsv_v(img_bgr)
    v_eq = cv2.equalizeHist(v)
    return _merge_v(hsv, v_eq)


def gamma_correction(img_bgr, gamma=0.5):
    """Fixed gamma correction V_out = 255 * (V/255)^gamma on the V channel.

    gamma < 1 brightens dark images. A single fixed value cannot adapt to
    scenes that contain both very dark and bright regions.
    """
    hsv, v = _to_hsv_v(img_bgr)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                   dtype=np.uint8)
    v_g = cv2.LUT(v, lut)
    return _merge_v(hsv, v_g)


# --------------------------------------------------------------------------- #
# Components of the proposed method (also used for the ablation study)
# --------------------------------------------------------------------------- #
def clahe_v(img_bgr, clip_limit=2.0, tile=8):
    """CLAHE applied to the V channel only."""
    hsv, v = _to_hsv_v(img_bgr)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile), int(tile)))
    v_c = clahe.apply(v)
    return _merge_v(hsv, v_c)


def single_scale_retinex_v(img_bgr, sigma=30.0):
    """Single Scale Retinex (SSR) applied to the V channel only.

    SSR(x) = log(V(x) + 1) - log( G_sigma * V (x) + 1 ),
    where G_sigma is a Gaussian surround. The result is min-max normalized
    back to the 0-255 range.
    """
    hsv, v = _to_hsv_v(img_bgr)
    v_f = v.astype(np.float64) + 1.0
    surround = cv2.GaussianBlur(v_f, (0, 0), sigmaX=float(sigma)) + 1.0
    retinex = np.log(v_f) - np.log(surround)
    v_r = _minmax_to_uint8(retinex)
    return _merge_v(hsv, v_r)


# --------------------------------------------------------------------------- #
# Proposed method
# --------------------------------------------------------------------------- #
def hybrid_clahe_retinex(img_bgr, clip_limit=4.0, sigma=30.0,
                         w_clahe=0.6, w_ssr=0.4, tile=8):
    """Proposed Hybrid CLAHE-Retinex with HSV Value-channel fusion.

    Pipeline
    --------
    1. RGB(BGR) -> HSV, extract V.
    2. CLAHE on V              -> V_clahe   (local contrast)
    3. SSR on V (parallel)     -> V_ssr     (reflectance / detail recovery)
    4. Weighted fusion         -> V_f = w_clahe * V_clahe + w_ssr * V_ssr
    5. Merge V_f with original H, S and convert back to BGR.

    Parameters
    ----------
    clip_limit : CLAHE contrast clipping limit.
    sigma      : Gaussian surround scale of the SSR stage.
    w_clahe, w_ssr : fusion weights (should sum to 1).
    tile       : CLAHE tile grid size (tile x tile).
    """
    hsv, v = _to_hsv_v(img_bgr)

    # Stage 1: CLAHE on V
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile), int(tile)))
    v_clahe = clahe.apply(v).astype(np.float64)

    # Stage 2: SSR on V (parallel branch)
    v_f = v.astype(np.float64) + 1.0
    surround = cv2.GaussianBlur(v_f, (0, 0), sigmaX=float(sigma)) + 1.0
    retinex = np.log(v_f) - np.log(surround)
    v_ssr = _minmax_to_uint8(retinex).astype(np.float64)

    # Stage 3: weighted fusion
    s = w_clahe + w_ssr
    w_clahe, w_ssr = w_clahe / s, w_ssr / s  # safeguard normalization
    v_fused = w_clahe * v_clahe + w_ssr * v_ssr

    return _merge_v(hsv, v_fused)


def hybrid_adaptive(img_bgr, clip_limit=4.0, sigma=30.0, k=0.35, tile=8):
    """Proposed Hybrid CLAHE-Retinex with CONTENT-ADAPTIVE fusion.

    Instead of fixed global weights, the per-pixel fusion weight is derived
    from a smoothed luminance (illumination) estimate. The rationale is that
    the two branches are reliable in complementary regions:

      * In DARK regions the SSR log-ratio strongly amplifies sensor noise, so
        the bounded-contrast CLAHE branch should dominate.
      * In well-lit / structured regions CLAHE adds little while SSR safely
        recovers reflectance detail, so SSR is allowed more weight.

    We estimate the local illumination as a Gaussian-smoothed luminance
    L(x,y) = (G_sigma * V)(x,y) / 255  in [0,1], and set the SSR weight to

        w_SSR(x,y) = L(x,y)^k ,   w_CLAHE(x,y) = 1 - w_SSR(x,y),

    where the exponent k>0 controls how quickly SSR is suppressed as the
    region darkens (larger k => SSR confined to brighter regions). The fused
    luminance is the per-pixel convex combination

        V_fused(x,y) = w_CLAHE(x,y) V_CLAHE(x,y) + w_SSR(x,y) V_SSR(x,y).

    This replaces the three hand-picked global weight pairs with a single
    interpretable parameter k that is tuned by the same metric-driven search.
    """
    hsv, v = _to_hsv_v(img_bgr)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile), int(tile)))
    v_clahe = clahe.apply(v).astype(np.float64)

    v_f = v.astype(np.float64) + 1.0
    surround = cv2.GaussianBlur(v_f, (0, 0), sigmaX=float(sigma)) + 1.0
    retinex = np.log(v_f) - np.log(surround)
    v_ssr = _minmax_to_uint8(retinex).astype(np.float64)

    # per-pixel illumination estimate in [0,1]
    illum = cv2.GaussianBlur(v.astype(np.float64), (0, 0),
                             sigmaX=float(sigma)) / 255.0
    w_ssr = np.clip(illum, 0, 1) ** float(k)   # bright -> more SSR
    w_clahe = 1.0 - w_ssr

    v_fused = w_clahe * v_clahe + w_ssr * v_ssr
    return _merge_v(hsv, v_fused)
METHODS = {
    "GHE": lambda img, **kw: global_histogram_equalization(img),
    "Gamma": lambda img, gamma=0.5, **kw: gamma_correction(img, gamma=gamma),
    "CLAHE-only": lambda img, clip_limit=4.0, tile=8, **kw:
        clahe_v(img, clip_limit=clip_limit, tile=tile),
    "SSR-only": lambda img, sigma=30.0, **kw:
        single_scale_retinex_v(img, sigma=sigma),
    "Fixed-fusion": lambda img, clip_limit=4.0, sigma=30.0,
        w_clahe=0.6, w_ssr=0.4, tile=8, **kw:
        hybrid_clahe_retinex(img, clip_limit=clip_limit, sigma=sigma,
                             w_clahe=w_clahe, w_ssr=w_ssr, tile=tile),
    "Proposed": lambda img, clip_limit=4.0, sigma=30.0,
        k=0.35, tile=8, **kw:
        hybrid_adaptive(img, clip_limit=clip_limit, sigma=sigma,
                        k=k, tile=tile),
}

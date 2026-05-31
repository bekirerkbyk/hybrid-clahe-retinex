"""
Main experiment driver.

Runs:
  1. Method comparison (GHE, Gamma, CLAHE-only, SSR-only, Proposed) over a set
     of clean reference images degraded with a synthetic low-light model and
     multiple noise realizations. Reports PSNR, SSIM, BRISQUE, entropy, time.
  2. Hyperparameter study for the proposed method:
       - CLAHE clip limit in {2, 4, 8}
       - SSR Gaussian sigma in {15, 30, 80}
       - fusion weights in {(0.5,0.5), (0.7,0.3), (0.6,0.4)}
  3. Ablation study (CLAHE-only vs SSR-only vs Proposed) -- derived from (1).

Outputs CSV files into ../results and saves the raw reference / low-light /
enhanced images needed by the figure script.

Author: Bekir Erakbiyik (2211051073)
"""

import os
import time
import csv
import cv2
import numpy as np
from skimage import data

import methods as M
import metrics as Q
from degradation import degrade_lowlight

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGDATA = os.path.join(HERE, "..", "figures", "_data")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDATA, exist_ok=True)

MAX_SIDE = 512
N_NOISE = 3          # noise realizations per image
SEEDS = list(range(N_NOISE))

# Default operating point of the proposed method
DEF = dict(clip_limit=4.0, sigma=30.0, w_clahe=0.6, w_ssr=0.4, k=0.35, tile=8)


def load_images():
    """Load natural color reference images.

    Uses a subset of the Kodak Lossless True Color Image Suite (standard
    benchmark of natural scenes). Falls back to skimage sample images if the
    Kodak files are not present.
    """
    kodak_dir = os.path.join(HERE, "..", "data", "kodak")
    imgs = {}
    if os.path.isdir(kodak_dir):
        names = ["kodim01", "kodim05", "kodim07", "kodim19", "kodim23"]
        for n in names:
            p = os.path.join(kodak_dir, n + ".png")
            bgr = cv2.imread(p)  # already BGR uint8
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            if max(h, w) > MAX_SIDE:
                s = MAX_SIDE / max(h, w)
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
            imgs[n] = bgr
    if imgs:
        return imgs
    # fallback
    from skimage import data
    for n in ["astronaut", "coffee", "cat", "rocket"]:
        rgb = getattr(data, n)()[:, :, :3]
        imgs[n] = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return imgs


def eval_method(name, ref, low, **params):
    """Apply a method to `low`, return metrics vs `ref`."""
    t0 = time.perf_counter()
    out = M.METHODS[name](low, **params)
    dt = (time.perf_counter() - t0) * 1000.0  # ms
    return out, dict(
        method=name,
        psnr=Q.psnr(ref, out),
        ssim=Q.ssim(ref, out),
        brisque=Q.brisque(out),
        entropy=Q.entropy(out),
        time_ms=dt,
    )


def run_comparison(imgs):
    rows = []
    for img_name, ref in imgs.items():
        for seed in SEEDS:
            low = degrade_lowlight(ref, seed=seed)
            # baseline "no enhancement" row (the degraded input itself)
            rows.append(dict(image=img_name, seed=seed, method="Low-light(input)",
                             psnr=Q.psnr(ref, low), ssim=Q.ssim(ref, low),
                             brisque=Q.brisque(low), entropy=Q.entropy(low),
                             time_ms=0.0))
            outs = {}
            for name in ["GHE", "Gamma", "CLAHE-only", "SSR-only",
                         "Fixed-fusion", "Proposed"]:
                out, m = eval_method(name, ref, low, **DEF)
                m.update(image=img_name, seed=seed)
                rows.append(m)
                outs[name] = out
            # save montage set for seed 0 of every image
            if seed == 0:
                d = os.path.join(FIGDATA, img_name)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, "ref.png"), ref)
                cv2.imwrite(os.path.join(d, "low.png"), low)
                for name, out in outs.items():
                    cv2.imwrite(os.path.join(d, f"{name}.png"), out)
    return rows


def run_hyperparams(imgs):
    """Sweep one factor at a time around the default operating point.

    Uses the full-reference fidelity metrics (PSNR, SSIM) as the optimization
    criteria; these are fast and directly quantify reconstruction quality.
    """
    rows = []

    # Pre-compute degraded inputs once (reused across all configurations)
    degraded = [(ref, degrade_lowlight(ref, seed=seed))
                for ref in imgs.values() for seed in SEEDS]

    def avg_metric(params):
        ps, ss = [], []
        for ref, low in degraded:
            out = M.hybrid_adaptive(low, **params)
            ps.append(Q.psnr(ref, out))
            ss.append(Q.ssim(ref, out))
        return np.mean(ps), np.mean(ss)

    base = dict(clip_limit=4.0, sigma=30.0, k=0.35, tile=8)

    # CLAHE clip limit
    for clip in [2.0, 4.0, 8.0]:
        p = dict(base); p["clip_limit"] = clip
        psnr_, ssim_ = avg_metric(p)
        rows.append(dict(factor="clip_limit", value=clip,
                         psnr=psnr_, ssim=ssim_))
    # SSR sigma
    for sig in [15.0, 30.0, 80.0]:
        p = dict(base); p["sigma"] = sig
        psnr_, ssim_ = avg_metric(p)
        rows.append(dict(factor="sigma", value=sig,
                         psnr=psnr_, ssim=ssim_))
    # adaptive fusion exponent k
    for kk in [0.2, 0.35, 0.5, 0.8]:
        p = dict(base); p["k"] = kk
        psnr_, ssim_ = avg_metric(p)
        rows.append(dict(factor="adaptive_k", value=kk,
                         psnr=psnr_, ssim=ssim_))
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows):
    """Mean +/- std per method across all images/seeds."""
    methods = ["Low-light(input)", "GHE", "Gamma", "CLAHE-only",
               "SSR-only", "Fixed-fusion", "Proposed"]
    summary = []
    for name in methods:
        sub = [r for r in rows if r["method"] == name]
        def col(k):
            vals = np.array([r[k] for r in sub], dtype=float)
            return np.nanmean(vals), np.nanstd(vals)
        pm, ps = col("psnr"); sm, ss = col("ssim")
        bm, bs = col("brisque"); em, es = col("entropy")
        tm, ts = col("time_ms")
        summary.append(dict(method=name,
                            psnr_mean=pm, psnr_std=ps,
                            ssim_mean=sm, ssim_std=ss,
                            brisque_mean=bm, brisque_std=bs,
                            entropy_mean=em, entropy_std=es,
                            time_ms_mean=tm, time_ms_std=ts))
    return summary


def main():
    print("Loading images...")
    imgs = load_images()
    print("Images:", {k: v.shape for k, v in imgs.items()})

    print("Running method comparison...")
    rows = run_comparison(imgs)
    write_csv(os.path.join(RESULTS, "per_sample.csv"), rows,
              ["image", "seed", "method", "psnr", "ssim", "brisque",
               "entropy", "time_ms"])

    summary = summarize(rows)
    write_csv(os.path.join(RESULTS, "summary.csv"), summary,
              ["method", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
               "brisque_mean", "brisque_std", "entropy_mean", "entropy_std",
               "time_ms_mean", "time_ms_std"])

    print("Running hyperparameter study...")
    hp = run_hyperparams(imgs)
    write_csv(os.path.join(RESULTS, "hyperparams.csv"), hp,
              ["factor", "value", "psnr", "ssim"])

    print("\n=== SUMMARY (mean over %d images x %d seeds) ===" %
          (len(imgs), N_NOISE))
    for s in summary:
        print(f"{s['method']:18s} PSNR={s['psnr_mean']:5.2f}  "
              f"SSIM={s['ssim_mean']:.3f}  BRISQUE={s['brisque_mean']:5.2f}  "
              f"Entropy={s['entropy_mean']:.2f}  t={s['time_ms_mean']:.1f}ms")
    print("\n=== HYPERPARAMETERS ===")
    for r in hp:
        print(f"{r['factor']:22s} {str(r['value']):8s} "
              f"PSNR={r['psnr']:5.2f} SSIM={r['ssim']:.3f}")
    print("\nDone. CSVs in results/, figure data in figures/_data/")


if __name__ == "__main__":
    main()

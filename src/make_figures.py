"""
Generate all figures for the paper from the experiment outputs.

Figures produced (saved to ../figures):
  fig_pipeline.png       : block diagram / graphical abstract of the method
  fig_qualitative.png    : reference, low-light, and all method outputs (montage)
  fig_metrics_bar.png    : PSNR / SSIM / BRISQUE bar charts (with std error bars)
  fig_hyperparams.png    : sensitivity of PSNR/SSIM to clip, sigma, fusion weights

Author: Bekir Erakbiyik (2211051073)
"""

import os
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
FIGDATA = os.path.join(FIG, "_data")
RESULTS = os.path.join(HERE, "..", "results")
plt.rcParams.update({"font.size": 11, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def bgr2rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.04,rounding_size=0.08",
                     linewidth=1.4, edgecolor="#222", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14,
                     linewidth=1.3, color="#333"))

    box(0.2, 2.5, 1.7, 1.0, "Input RGB\n(low-light)", "#e8eef7")
    box(2.3, 2.5, 1.7, 1.0, "RGB->HSV\nextract V", "#e8eef7")
    box(4.5, 4.0, 1.9, 1.0, "CLAHE on V\n(local contrast)", "#dff0d8")
    box(4.5, 1.0, 1.9, 1.0, "SSR on V\n(reflectance)", "#fcf0d8")
    box(4.5, 2.5, 1.9, 0.9, "Illumination\nL = G_s * V", "#ede7f6")
    box(6.9, 2.5, 1.5, 1.0, "Adaptive\nFusion", "#f5d9e0")
    box(8.6, 2.5, 1.3, 1.0, "HSV->RGB\nOutput", "#e8eef7")

    arrow(1.9, 3.0, 2.3, 3.0)
    arrow(4.0, 3.2, 4.5, 4.5)
    arrow(4.0, 3.0, 4.5, 2.95)
    arrow(4.0, 2.8, 4.5, 1.5)
    arrow(6.4, 4.5, 6.9, 3.2)
    arrow(6.4, 1.5, 6.9, 2.8)
    arrow(6.4, 2.95, 6.9, 3.0)
    arrow(8.4, 3.0, 8.6, 3.0)
    ax.text(7.65, 2.15, r"$w_S=L^{k}$", ha="center", fontsize=9, color="#444")
    ax.set_title("Proposed Hybrid CLAHE-Retinex Pipeline "
                 "(content-adaptive V-channel fusion)", fontsize=11.5)
    fig.savefig(os.path.join(FIG, "fig_pipeline.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_qualitative(img_name="kodim23", out="fig_qualitative.png"):
    order = [("ref", "Reference (clean)"), ("low", "Low-light input"),
             ("GHE", "GHE"), ("Gamma", "Gamma"), ("CLAHE-only", "CLAHE-only"),
             ("SSR-only", "SSR-only"), ("Fixed-fusion", "Fixed fusion"),
             ("Proposed", "Proposed (adaptive)")]
    persamp = read_csv(os.path.join(RESULTS, "per_sample.csv"))
    look = {}
    for r in persamp:
        if r["image"] == img_name and r["seed"] == "0":
            look[r["method"]] = (float(r["psnr"]), float(r["ssim"]))

    d = os.path.join(FIGDATA, img_name)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.08})
    axes = axes.ravel()
    keymap = {"ref": None, "low": "Low-light(input)", "GHE": "GHE",
              "Gamma": "Gamma", "CLAHE-only": "CLAHE-only",
              "SSR-only": "SSR-only", "Fixed-fusion": "Fixed-fusion",
              "Proposed": "Proposed"}
    for ax, (key, title) in zip(axes, order):
        ax.imshow(bgr2rgb(os.path.join(d, f"{key}.png")))
        ax.set_xticks([]); ax.set_yticks([])
        mkey = keymap[key]
        if mkey in look:
            p, s = look[mkey]
            title += f"\nPSNR={p:.2f} | SSIM={s:.3f}"
        ax.set_title(title, fontsize=9.5, pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor("#bbb")
    fig.suptitle("Qualitative comparison on a synthetically degraded image "
                 f"({img_name})", fontsize=12, y=0.98)
    fig.savefig(os.path.join(FIG, out))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_metrics_bar():
    s = read_csv(os.path.join(RESULTS, "summary.csv"))
    methods = [r["method"] for r in s]
    colors = ["#9aa0a6", "#4c78a8", "#f58518", "#54a24b", "#b279a2",
              "#9d755d", "#e45756"]

    def get(k):
        return np.array([float(r[k]) for r in s])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [("psnr_mean", "psnr_std", "PSNR (dB) - higher better"),
             ("ssim_mean", "ssim_std", "SSIM - higher better"),
             ("brisque_mean", "brisque_std", "BRISQUE - lower better")]
    for ax, (mk, sk, title) in zip(axes, specs):
        vals, errs = get(mk), get(sk)
        bars = ax.bar(range(len(methods)), vals, yerr=errs, capsize=3,
                      color=colors, edgecolor="#333", linewidth=0.6)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        # highlight Proposed
        bars[methods.index("Proposed")].set_edgecolor("k")
        bars[methods.index("Proposed")].set_linewidth(1.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_metrics_bar.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_hyperparams():
    hp = read_csv(os.path.join(RESULTS, "hyperparams.csv"))
    def sub(f):
        rows = [r for r in hp if r["factor"] == f]
        return ([r["value"] for r in rows],
                [float(r["psnr"]) for r in rows],
                [float(r["ssim"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("clip_limit", "CLAHE clip limit"),
              ("sigma", "SSR Gaussian sigma"),
              ("adaptive_k", "Adaptive fusion exponent k")]
    for ax, (f, label) in zip(axes, panels):
        x, p, s = sub(f)
        xs = range(len(x))
        ax.plot(xs, p, "o-", color="#e45756", label="PSNR (dB)")
        ax.set_xticks(list(xs)); ax.set_xticklabels(x)
        ax.set_xlabel(label); ax.set_ylabel("PSNR (dB)", color="#e45756")
        ax.tick_params(axis="y", labelcolor="#e45756")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(xs, s, "s--", color="#4c78a8", label="SSIM")
        ax2.set_ylabel("SSIM", color="#4c78a8")
        ax2.tick_params(axis="y", labelcolor="#4c78a8")
    fig.suptitle("Hyperparameter sensitivity of the proposed method "
                 "(mean over 5 images x 3 noise realizations)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "fig_hyperparams.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_qualitative("kodim23", "fig_qualitative.png")
    fig_qualitative("kodim05", "fig_qualitative2.png")
    fig_qualitative("kodim19", "fig_qualitative3.png")
    fig_metrics_bar()
    fig_hyperparams()
    print("Figures written to", FIG)
    print(os.listdir(FIG))

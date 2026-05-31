# Hybrid CLAHE–Retinex Enhancement for Nighttime Surveillance Images

Source code for the term project / paper **"Hybrid CLAHE–Retinex Enhancement
for Nighttime Surveillance Images"** by Bekir Erakbıyık (Student ID:
2211051073).

The method enhances low-light images by fusing **CLAHE** (local contrast) and
**Single Scale Retinex / SSR** (reflectance recovery) on the **Value (V)
channel** of the HSV color space, leaving Hue and Saturation untouched so the
original color is preserved. It is training-free, runs on CPU in linear time,
and is compared against Global Histogram Equalization (GHE) and fixed Gamma
correction, with an ablation against the CLAHE-only and SSR-only branches.

## Method overview

```
Input RGB ─▶ RGB→HSV ─▶ extract V ─┬─▶ CLAHE on V ─────────┐
                                   ├─▶ SSR on V ────────────┤─▶ adaptive ─▶ merge H,S ─▶ Output RGB
                                   └─▶ illumination L=Gσ*V ─┘   fusion
```

The two branches are applied **independently** to the V channel and then
combined by a **content-adaptive** per-pixel fusion (not a fixed global blend):

```
L(x,y)       = (Gσ * V)(x,y) / 255        # local illumination in [0,1]
w_SSR(x,y)   = L(x,y)^k                    # bright regions -> more SSR
w_CLAHE(x,y) = 1 - w_SSR(x,y)             # dark regions -> more CLAHE (bounded noise)
V_fused      = w_CLAHE · V_CLAHE + w_SSR · V_SSR
```

Defaults: CLAHE clip limit `β = 4.0`, SSR Gaussian `σ = 30`, adaptive exponent
`k = 0.35`. A fixed-weight fusion (`w_C = 0.6, w_S = 0.4`) is kept as a baseline
to isolate the benefit of adaptivity.

## Repository structure

```
.
├── src/
│   ├── methods.py          # GHE, Gamma, CLAHE, SSR, and the proposed hybrid
│   ├── degradation.py      # synthetic low-light degradation model
│   ├── metrics.py          # PSNR, SSIM, BRISQUE, entropy
│   ├── run_experiments.py  # comparison + hyperparameter + ablation experiments
│   └── make_figures.py     # generates all paper figures
├── data/kodak/             # 5 Kodak reference images (downloaded)
├── results/                # CSV outputs (summary, per-sample, hyperparams)
├── figures/                # generated PNG figures
├── paper/                  # IEEE LaTeX article (main.tex, references.bib)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
# optional, for the no-reference BRISQUE metric:
pip install image-quality
```

Tested with Python 3.12, OpenCV 4.13, scikit-image 0.26.

## Reproducing the results

```bash
cd src
python run_experiments.py     # writes CSVs to ../results and montage data to ../figures/_data
python make_figures.py        # writes fig_*.png to ../figures
```

> Note: BRISQUE (`image-quality`) is slow (~1.5 s/image). The method comparison
> uses it; the hyperparameter sweep uses the fast PSNR/SSIM metrics only.

## Dataset and evaluation protocol

Full-reference metrics (PSNR, SSIM) need a clean ground truth, which raw
low-light captures do not provide. We therefore use the standard **synthetic
degradation** protocol: clean Kodak images are darkened (gamma + brightness
scaling) and corrupted with signal-dependent Poisson shot noise plus Gaussian
read noise, then enhanced and compared against the clean reference. Five Kodak
images × three noise realizations = 15 samples.

## Main results (mean over 15 samples)

| Method            | PSNR ↑ | SSIM ↑ | BRISQUE ↓ | Time (ms) |
|-------------------|:------:|:------:|:---------:|:---------:|
| Low-light (input) | 10.14  | 0.290  | 22.8      | –         |
| GHE               | 17.09  | 0.454  | 49.9      | 1.8       |
| Gamma             | 15.50  | 0.513  | 45.3      | 1.8       |
| CLAHE-only        | 15.55  | 0.483  | 47.5      | 2.8       |
| SSR-only          | 15.70  | 0.423  | 55.8      | 40.0      |
| Fixed-fusion      | 18.39  | 0.494  | 51.4      | 41.5      |
| **Proposed (adaptive)** | **19.03** | 0.494 | 51.7 | 77.9 |

The content-adaptive fusion gives the highest PSNR — beating the otherwise
identical fixed-weight fusion by +0.64 dB and ranking first on every individual
Kodak image — while preserving natural color through the HSV design.

## Building the paper

```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # second pass for cross-references
```

Produces a 6-page IEEE A4 conference article (`main.pdf`).

## License

MIT License (see below). Kodak images are used for research/benchmarking
purposes and remain the property of their original copyright holders.

## Citation

If you use this code, please cite the accompanying report and this repository:

```
B. Erakbıyık, "Hybrid CLAHE–Retinex Enhancement for Nighttime Surveillance
Images," term project, 2026. https://github.com/bekirerkbyk/hybrid-clahe-retinex
```

## Acknowledgement

Generative AI tools (ChatGPT and Gemini) were used to help structure the report
and improve writing clarity. All code, experiments, and results were produced
and verified by the author.

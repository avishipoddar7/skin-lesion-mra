"""Report figures (PRD FR-17). Colours follow the PRD palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, TEAL, CORAL = "#1F3A5F", "#2A9D8F", "#E76F51"


def _show(ax, img, title, cmap=None):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def stretch(a):
    a = np.abs(np.asarray(a, np.float32))
    return a / (np.percentile(a, 99) + 1e-8)


def pipeline_montage(results, titles):
    """One row per analysed image: original, hair mask, hair removed, LL, coarse, final, overlay."""
    fig, axes = plt.subplots(len(results), 7, figsize=(17, 2.5 * len(results)))
    axes = np.atleast_2d(axes)
    names = ["Original", "Hair mask", "Hair removed", "LL band", "Coarse mask",
             "Final mask", "Contour overlay"]
    for row, (res, title) in enumerate(zip(results, titles)):
        im, lv = res["images"], res["meta"]["level"]
        panels = [im["original"], im["hair_mask"], im["hair_removed"],
                  im["subbands"][lv]["LL"], im["coarse_mask"], im["final_mask"], im["overlay"]]
        for col, (p, n) in enumerate(zip(panels, names)):
            _show(axes[row, col], p, f"{title}\n{n}" if col == 0 else n,
                  cmap="gray" if col in (1, 3, 4, 5) else None)
    fig.tight_layout()
    return fig


def subband_grid(res):
    """LL / LH(H) / HL(V) / HH(D) for each level of one image."""
    sb = res["images"]["subbands"]
    fig, axes = plt.subplots(len(sb), 4, figsize=(9, 2.3 * len(sb)))
    axes = np.atleast_2d(axes)
    for r, k in enumerate(sorted(sb)):
        for c, (key, name) in enumerate((("LL", "LL"), ("H", "LH"), ("V", "HL"), ("D", "HH"))):
            band = sb[k][key]
            _show(axes[r, c], band if key == "LL" else stretch(band), f"Level {k}: {name}", "gray")
    fig.suptitle(f"{res['meta']['wavelet']} DWT subbands", fontsize=11)
    fig.tight_layout()
    return fig


def pyramid_figure(res):
    g, lap = res["images"]["gaussian_pyramid"], res["images"]["laplacian_pyramid"]
    fig, axes = plt.subplots(2, len(g), figsize=(2.6 * len(g), 5.4))
    for i, a in enumerate(g):
        _show(axes[0, i], a, f"Gaussian {i} ({a.shape[1]}px)", "gray")
    for i, a in enumerate(lap):
        _show(axes[1, i], a if i == len(lap) - 1 else stretch(a), f"Laplacian {i}", "gray")
    fig.tight_layout()
    return fig


def dice_comparison(exp):
    """Grouped bars: mean Dice by wavelet and level, with and without hair removal."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, hair in zip(axes, (True, False)):
        d = exp[exp.hair_removal == hair]
        w = 0.25
        for i, wav in enumerate(("haar", "db4", "sym4")):
            s = d[d.wavelet == wav].sort_values("level")
            ax.bar(s.level + (i - 1) * w, s.dice, w, label=wav, color=[NAVY, TEAL, CORAL][i])
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Decomposition level")
        ax.set_title("Hair removal ON" if hair else "Hair removal OFF")
        ax.set_ylim(0.5, 0.9)
        ax.axhline(0.8, ls="--", c="grey", lw=1)
    axes[0].set_ylabel("Mean Dice")
    axes[0].legend()
    fig.tight_layout()
    return fig


def confusion_figure(cm):
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Non-melanoma", "Melanoma"])
    ax.set_yticks([0, 1], ["Non-melanoma", "Melanoma"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (5-fold CV)")
    fig.tight_layout()
    return fig


def roc_figure(y, scores, auc):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, scores)
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot(fpr, tpr, c=NAVY, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (5-fold CV)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def energy_by_class(feats):
    """Mean detail-subband energy (log scale) for non-melanoma vs melanoma."""
    cols = [c for c in feats.columns if c.endswith("_energy")]
    m = feats.groupby("label")[cols].mean()
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.2, m.loc[0], 0.4, color=TEAL, label="Non-melanoma")
    ax.bar(x + 0.2, m.loc[1], 0.4, color=CORAL, label="Melanoma")
    ax.set_xticks(x, [c.replace("_energy", "") for c in cols])
    ax.set_yscale("log")
    ax.set_ylabel("Mean energy (log)")
    ax.legend()
    fig.tight_layout()
    return fig

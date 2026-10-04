"""Stage 3: lesion segmentation = Otsu on the DWT approximation (LL) band + morphology."""
import cv2
import numpy as np
from scipy import ndimage as ndi

import config
from src import wavelet_utils
from src.errors import NoLesionError


def _normalise(a):
    a = a.astype(np.float32)
    return ((a - a.min()) / (a.max() - a.min() + 1e-8) * 255).astype(np.uint8)


def _select_component(mask, penalty):
    """Keep the connected component with the best score. The score is its area, reduced
    by `penalty` for every fraction of the image border it touches (vignetting)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape
    best, best_score = 0, -1.0
    for i in range(1, n):
        comp = labels == i
        border = (comp[0].sum() + comp[-1].sum() + comp[:, 0].sum() + comp[:, -1].sum())
        touch = min(1.0, border / (2 * (h + w)) * 8)
        score = stats[i, cv2.CC_STAT_AREA] * (1 - penalty * 4 * touch)
        if score > best_score:
            best, best_score = i, score
    return labels == best


def suppress_vignette(gray):
    """Fill the dark circular dermoscope border (dark regions touching the image edge)
    with the median grey level, so Otsu does not mistake it for lesion (PRD FR-07)."""
    g = gray.astype(np.float32)
    med = float(np.median(g))
    dark = (g < config.VIGNETTE_RATIO * med).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(dark, 8)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    vig = np.isin(labels, edge[edge > 0]).astype(np.uint8)
    if config.VIGNETTE_GROW:
        vig = cv2.dilate(vig, np.ones((config.VIGNETTE_GROW,) * 2, np.uint8))
    out = g.copy()
    out[vig > 0] = med
    return out


def segment_lesion(gray, wavelet=config.WAVELET, level=config.LEVEL, strict=True):
    """Returns (mask, info). mask: bool array like `gray`. info holds the LL band,
    the coarse mask and the Otsu threshold. With strict=True (the app) an implausible
    mask raises NoLesionError; batch jobs pass strict=False and keep the best-effort mask."""
    gray = suppress_vignette(gray)
    coeffs = wavelet_utils.decompose(gray, wavelet, level)
    ll = coeffs[0]
    ll8 = cv2.GaussianBlur(_normalise(ll), (config.BLUR_KSIZE,) * 2, 0)
    thr, coarse = cv2.threshold(ll8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    up = cv2.resize(coarse, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = up > 127
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.OPEN_SIZE,) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.CLOSE_SIZE,) * 2)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m = ndi.binary_fill_holes(m)
    m = _select_component(m, config.BORDER_PENALTY)
    m = ndi.binary_fill_holes(m)

    frac = m.mean()
    if strict and (frac < config.MIN_LESION_FRAC or frac > config.MAX_LESION_FRAC):
        raise NoLesionError("No lesion could be found in this image.")
    if m.sum() == 0:
        raise NoLesionError("No lesion could be found in this image.")
    return m, {"LL": ll, "coarse_mask": coarse > 127, "otsu_threshold": float(thr)}

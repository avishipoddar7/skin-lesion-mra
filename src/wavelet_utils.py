"""Stage 2: 2-D DWT (Haar / db4 / sym4) and Gaussian / Laplacian pyramids."""
import cv2
import numpy as np
import pywt


def decompose(gray, wavelet, level):
    """Multilevel 2-D DWT. Returns pywt's [LL_n, (H_n, V_n, D_n), ..., (H_1, V_1, D_1)]."""
    return pywt.wavedec2(gray.astype(np.float32), wavelet, level=level)


def subband_levels(gray, wavelet, level):
    """Level-by-level decomposition for display and features.
    Returns {k: {"LL", "H", "V", "D"}} for k = 1..level, where level 1 is the finest
    scale and each LL is the input of the next level. H/V/D correspond to
    LH/HL/HH in the textbook naming."""
    out, ll = {}, gray.astype(np.float32)
    for k in range(1, level + 1):
        ll, (h, v, d) = pywt.dwt2(ll, wavelet)
        out[k] = {"LL": ll, "H": h, "V": v, "D": d}
    return out


def gaussian_pyramid(gray, levels=3):
    """[level 0 (original), level 1, ...] with cv2.pyrDown."""
    pyr = [gray.astype(np.float32)]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian_pyramid(gray, levels=3):
    """Band-pass levels (G_i - up(G_{i+1})) followed by the coarsest Gaussian level."""
    g = gaussian_pyramid(gray, levels)
    lap = []
    for i in range(levels):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap

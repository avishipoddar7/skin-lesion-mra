"""Stage 4: wavelet texture features and ABCD clinical features."""
import cv2
import numpy as np
from skimage import measure

import config
from src import wavelet_utils


def _energy_entropy(band):
    c2 = np.square(band.astype(np.float64))
    energy = float(c2.mean())
    total = c2.sum()
    if total <= 0:
        return energy, 0.0
    p = c2.ravel() / total
    p = p[p > 0]
    return energy, float(-(p * np.log2(p)).sum())


def _crop_around(img, mask, size):
    """size x size window centred on the lesion centroid, shifted to stay inside."""
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = img.shape[:2]
    size = min(size, h, w)
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return img[y0:y0 + size, x0:x0 + size]


def wavelet_features(gray, mask, wavelet, level):
    """Energy and entropy of each detail subband per level, plus LL mean/std,
    on a crop around the lesion."""
    crop = _crop_around(gray, mask, config.CROP_SIZE)
    bands = wavelet_utils.subband_levels(crop, wavelet, level)
    f = {}
    for k, b in bands.items():
        for name in ("H", "V", "D"):
            e, h = _energy_entropy(b[name])
            f[f"L{k}_{name}_energy"] = e
            f[f"L{k}_{name}_entropy"] = h
    ll = bands[level]["LL"]
    f["LL_mean"] = float(ll.mean())
    f["LL_std"] = float(ll.std())
    return f


def _asymmetry(mask):
    """XOR area after flipping about each principal axis, / lesion area, averaged."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov((pts - c).T))
    area = float(mask.sum())
    h, w = mask.shape
    vals = []
    for axis in range(2):
        v = vecs[:, axis]
        d = pts - c
        along = d @ v
        perp = d - np.outer(along, v)
        mirrored = c + perp - np.outer(along, v)  # reflect across the other axis
        mx = np.rint(mirrored[:, 0]).astype(int)
        my = np.rint(mirrored[:, 1]).astype(int)
        flipped = np.zeros_like(mask, dtype=bool)
        ok = (mx >= 0) & (mx < w) & (my >= 0) & (my < h)
        flipped[my[ok], mx[ok]] = True
        flipped = cv2.morphologyEx(flipped.astype(np.uint8), cv2.MORPH_CLOSE,
                                   np.ones((3, 3), np.uint8)) > 0
        vals.append((mask ^ flipped).sum() / area)
    return float(np.mean(vals))


def abcd_features(rgb, mask):
    """A: asymmetry. B: compactness, solidity. C: colour mean/std per channel.
    D: relative major-axis length (fraction of image size)."""
    props = measure.regionprops(mask.astype(np.uint8))[0]
    area, per = float(props.area), float(props.perimeter)
    f = {
        "asymmetry": _asymmetry(mask),
        "border_ci": float(per ** 2 / (4 * np.pi * area)) if area else 0.0,
        "solidity": float(props.solidity),
    }
    for i, ch in enumerate("RGB"):
        px = rgb[..., i][mask].astype(np.float64)
        f[f"color_{ch}_mean"] = float(px.mean())
        f[f"color_{ch}_std"] = float(px.std())
    f["rel_diameter"] = float(props.axis_major_length / mask.shape[0])
    return f


def extract_features(rgb, gray, mask, wavelet=config.WAVELET, level=config.LEVEL):
    """Ordered dict: ABCD features then wavelet features."""
    f = abcd_features(rgb, mask)
    f.update(wavelet_features(gray, mask, wavelet, level))
    return f

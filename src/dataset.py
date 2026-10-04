"""PH2 loader: image paths, mask paths and diagnosis labels."""
import cv2
import numpy as np

import config

CLINICAL_NAMES = {0: "Common nevus", 1: "Atypical nevus", 2: "Melanoma"}


def _parse_labels():
    """Parse PH2_dataset.txt. The histological column is empty, so cells are
    split on '||' and empty cells are kept (dropping them shifts columns)."""
    labels = {}
    with open(config.PH2_LABELS, encoding="latin-1") as f:
        for line in f:
            if not line.strip().startswith("|| IMD"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("||")]
            labels[cells[0]] = int(cells[2])
    return labels


def load_index():
    """Return a sorted list of dicts: id, image_path, mask_path, clinical, label.
    label: melanoma = 1, common/atypical nevus = 0."""
    labels = _parse_labels()
    records = []
    for case, clinical in sorted(labels.items()):
        base = config.PH2_IMAGES / case
        records.append({
            "id": case,
            "image_path": base / f"{case}_Dermoscopic_Image" / f"{case}.bmp",
            "mask_path": base / f"{case}_lesion" / f"{case}_lesion.bmp",
            "clinical": clinical,
            "label": int(clinical == 2),
        })
    return records


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def load_prepared(records=None):
    """Yield (record, rgb256, mask256) for every case, using the same geometry as the app."""
    from src import preprocessing as pp
    for r in (records if records is not None else load_index()):
        yield r, pp.prepare(read_rgb(r["image_path"])), pp.prepare_mask(read_mask(r["mask_path"]))

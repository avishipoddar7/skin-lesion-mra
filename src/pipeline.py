"""The single analysis pipeline used by training, experiments and the app (PRD 8.2).

analyse() runs two paths:
  * display path    - the user's wavelet / level / hair settings (images shown in the UI)
  * prediction path - the model's own training configuration (feature names depend on
                      the level, so a model can only score features made with its config)
When the two configurations match, the work is done once."""
import time

import cv2
import numpy as np

import config
from src import classifier, features, preprocessing, segmentation, wavelet_utils


def process(rgb256, wavelet, level, hair_removal, strict=True):
    """Hair removal -> grayscale -> DWT -> segmentation -> features for one prepared image."""
    t = {}
    t0 = time.perf_counter()
    if hair_removal:
        clean, hair_mask = preprocessing.remove_hair(rgb256)
    else:
        clean, hair_mask = rgb256, np.zeros(rgb256.shape[:2], bool)
    t["hair_removal"] = time.perf_counter() - t0
    gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    t0 = time.perf_counter()
    mask, info = segmentation.segment_lesion(gray, wavelet, level, strict=strict)
    t["segmentation"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    feats = features.extract_features(clean, gray, mask, wavelet, level)
    t["features"] = time.perf_counter() - t0
    return {"clean": clean, "hair_mask": hair_mask, "gray": gray, "mask": mask,
            "info": info, "features": feats, "timings": t}


def overlay_contour(rgb, mask, color=(231, 111, 81)):
    out = rgb.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def analyse(image, wavelet=config.WAVELET, level=config.LEVEL,
            hair_removal=config.HAIR_REMOVAL, model=None, strict=True):
    """image: RGB uint8 array (any size >= 128 px). model: dict from train.py or None.
    Raises NoLesionError (via segmentation) when strict and no plausible lesion is found."""
    t_start = time.perf_counter()
    rgb = preprocessing.prepare(image)
    disp = process(rgb, wavelet, level, hair_removal, strict=strict)

    prediction, differs, model_cfg = None, False, None
    if model is not None:
        model_cfg = model["config"]
        same = (model_cfg["wavelet"], model_cfg["level"], model_cfg["hair_removal"]) == \
               (wavelet, level, hair_removal)
        differs = not same
        pred_feats = disp["features"] if same else process(
            rgb, model_cfg["wavelet"], model_cfg["level"], model_cfg["hair_removal"],
            strict=False)["features"]
        x = [pred_feats[n] for n in model["feature_names"]]
        label, score = classifier.predict(model["pipeline"], x)
        prediction = {"label": label, "score": score,
                      "threshold": config.DECISION_THRESHOLD}

    gray = disp["gray"]
    subbands = wavelet_utils.subband_levels(gray, wavelet, level)
    return {
        "images": {
            "original": rgb, "hair_removed": disp["clean"], "hair_mask": disp["hair_mask"],
            "subbands": subbands, "coarse_mask": disp["info"]["coarse_mask"],
            "final_mask": disp["mask"], "overlay": overlay_contour(disp["clean"], disp["mask"]),
            "gaussian_pyramid": wavelet_utils.gaussian_pyramid(gray, 3),
            "laplacian_pyramid": wavelet_utils.laplacian_pyramid(gray, 3),
        },
        "features": disp["features"],
        "prediction": prediction,
        "meta": {"wavelet": wavelet, "level": level, "hair_removal": hair_removal,
                 "time_ms": int((time.perf_counter() - t_start) * 1000),
                 "timings": {k: int(v * 1000) for k, v in disp["timings"].items()},
                 "otsu_threshold": disp["info"]["otsu_threshold"],
                 "model_config": model_cfg, "prediction_config_differs": differs},
    }


def dice_vs_expert(result, expert_mask):
    """Dice of the final mask against an expert mask given at the original image size."""
    from src import metrics
    return float(metrics.dice(result["images"]["final_mask"],
                              preprocessing.prepare_mask(expert_mask)))

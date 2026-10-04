"""Generate report figures and the demo sample images."""
import json
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd

import config
from src import dataset, metrics, pipeline, visualize

warnings.filterwarnings("ignore")
F = config.FIGURES_DIR


def main():
    F.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.MODEL_PATH)
    idx = {r["id"]: r for r in dataset.load_index()}

    # Pick one clearly segmented, correctly classified sample per class (training-set images)
    feats = pd.read_csv(config.RESULTS_DIR / "features_default.csv")
    picks, chosen = {}, {}
    for rec, rgb, mask in dataset.load_prepared():
        res = pipeline.analyse(dataset.read_rgb(rec["image_path"]), model=model, strict=False)
        d = metrics.dice(res["images"]["final_mask"], mask)
        ok = (res["prediction"]["label"] == "Melanoma") == bool(rec["label"])
        if ok and d > 0.88 and rec["clinical"] not in picks:
            picks[rec["clinical"]] = (rec, res, d)
    for c, (rec, res, d) in sorted(picks.items()):
        name = dataset.CLINICAL_NAMES[c].lower().replace(" ", "_")
        img = dataset.read_rgb(rec["image_path"])
        mask = dataset.read_mask(rec["mask_path"])
        cv2.imwrite(str(config.SAMPLES_DIR / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(config.SAMPLES_DIR / f"{name}_mask.png"), mask.astype(np.uint8) * 255)
        chosen[name] = {"id": rec["id"], "clinical": dataset.CLINICAL_NAMES[c],
                        "label": rec["label"], "dice": round(d, 3)}
    (config.SAMPLES_DIR / "samples.json").write_text(json.dumps(chosen, indent=2))
    print("samples:", chosen)

    # Pipeline montage and subbands on the chosen samples
    results, titles = [], []
    for name, info in chosen.items():
        res = pipeline.analyse(cv2.cvtColor(cv2.imread(str(config.SAMPLES_DIR / f"{name}.png")),
                                            cv2.COLOR_BGR2RGB), model=model)
        results.append(res)
        titles.append(info["clinical"])
    visualize.pipeline_montage(results, titles).savefig(F / "pipeline_montage.png", dpi=110)
    visualize.subband_grid(results[0]).savefig(F / "subband_grid.png", dpi=110)
    visualize.pyramid_figure(results[0]).savefig(F / "pyramids.png", dpi=110)

    # Experiment figures
    exp = pd.read_csv(config.RESULTS_DIR / "experiments.csv")
    visualize.dice_comparison(exp).savefig(F / "dice_comparison.png", dpi=110)
    cv = json.loads((config.RESULTS_DIR / "cv_default.json").read_text())
    scores = np.load(config.RESULTS_DIR / "cv_scores_default.npy")
    y = feats["label"].values
    pred = (scores >= config.DECISION_THRESHOLD).astype(int)
    cm = np.array([[((y == a) & (pred == b)).sum() for b in (0, 1)] for a in (0, 1)])
    visualize.confusion_figure(cm).savefig(F / "confusion_matrix.png", dpi=130)
    visualize.roc_figure(y, scores, cv["auc"]).savefig(F / "roc_curve.png", dpi=130)
    visualize.energy_by_class(feats).savefig(F / "subband_energy_by_class.png", dpi=130)
    print("figures written to", F)


if __name__ == "__main__":
    main()

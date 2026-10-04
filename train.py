"""Train the final model on the fixed default configuration and report cross-validated
performance. Writes models/model.pkl, results/features_default.csv, results/cv_default.json."""
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import config
from src import classifier, dataset, evaluation, metrics, pipeline

warnings.filterwarnings("ignore")


def build_feature_table(wavelet, level, hair_removal):
    rows, dices = [], []
    for rec, rgb, mask in dataset.load_prepared():
        out = pipeline.process(rgb, wavelet, level, hair_removal, strict=False)
        rows.append({"id": rec["id"], "label": rec["label"], **out["features"]})
        dices.append((metrics.dice(out["mask"], mask), metrics.jaccard(out["mask"], mask)))
    return pd.DataFrame(rows), np.array(dices)


def main():
    cfg = {"wavelet": config.WAVELET, "level": config.LEVEL,
           "hair_removal": config.HAIR_REMOVAL}
    df, dj = build_feature_table(**cfg)
    names = [c for c in df.columns if c not in ("id", "label")]
    X, y = df[names].values, df["label"].values
    df.to_csv(config.RESULTS_DIR / "features_default.csv", index=False)

    cv = evaluation.cv_evaluate(X, y)
    summary = {k: v for k, v in cv.items() if k != "first"}
    summary.update(dice_mean=float(dj[:, 0].mean()), jaccard_mean=float(dj[:, 1].mean()),
                   config=cfg, threshold=config.DECISION_THRESHOLD)
    (config.RESULTS_DIR / "cv_default.json").write_text(json.dumps(summary, indent=2))
    np.save(config.RESULTS_DIR / "cv_scores_default.npy", cv["first"]["scores"])

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": classifier.train(X, y), "feature_names": names, "config": cfg},
                config.MODEL_PATH)
    print(json.dumps(summary, indent=2))
    print("saved", config.MODEL_PATH)


if __name__ == "__main__":
    main()

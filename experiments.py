"""Experiment grid (E1-E4): 3 wavelets x 3 levels x hair removal on/off, plus the
feature-set ablation. The grid is a comparison only; the final configuration is fixed
in config.py. Writes results/experiments.csv and results/feature_ablation.csv."""
import time
import warnings

import pandas as pd

import config
from src import dataset, evaluation, metrics, pipeline

warnings.filterwarnings("ignore")

ABCD_PREFIXES = ("asymmetry", "border_ci", "solidity", "color_", "rel_diameter")


def run_config(prepared, wavelet, level, hair):
    rows, dj = [], []
    for rec, rgb, mask in prepared:
        out = pipeline.process(rgb, wavelet, level, hair, strict=False)
        rows.append({"label": rec["label"], **out["features"]})
        dj.append((metrics.dice(out["mask"], mask), metrics.jaccard(out["mask"], mask)))
    df = pd.DataFrame(rows)
    return df, pd.DataFrame(dj, columns=["dice", "jaccard"])


def main():
    prepared = list(dataset.load_prepared())
    grid, t0 = [], time.time()
    default_df = None
    for hair in (True, False):
        for wavelet in config.WAVELETS:
            for level in config.LEVELS:
                df, dj = run_config(prepared, wavelet, level, hair)
                X, y = df.drop(columns="label").values, df["label"].values
                cv = evaluation.cv_evaluate(X, y)
                grid.append({"wavelet": wavelet, "level": level, "hair_removal": hair,
                             "dice": dj.dice.mean(), "jaccard": dj.jaccard.mean(),
                             **{k: v for k, v in cv.items() if k != "first"}})
                if (wavelet, level, hair) == (config.WAVELET, config.LEVEL, config.HAIR_REMOVAL):
                    default_df = df
                print(f"{wavelet:5s} L{level} hair={hair!s:5s} dice={dj.dice.mean():.3f} "
                      f"auc={cv['auc']:.3f} sens={cv['sensitivity']:.2f}", flush=True)
    pd.DataFrame(grid).round(4).to_csv(config.RESULTS_DIR / "experiments.csv", index=False)

    # E4: feature-set ablation on the default configuration
    names = [c for c in default_df.columns if c != "label"]
    sets = {"wavelet only": [n for n in names if not n.startswith(ABCD_PREFIXES)],
            "ABCD only": [n for n in names if n.startswith(ABCD_PREFIXES)],
            "combined": names}
    y = default_df["label"].values
    abl = []
    for name, cols in sets.items():
        cv = evaluation.cv_evaluate(default_df[cols].values, y)
        abl.append({"feature_set": name, "n_features": len(cols),
                    **{k: v for k, v in cv.items() if k != "first"}})
        print(name, round(cv["auc"], 3), round(cv["sensitivity"], 2), flush=True)
    pd.DataFrame(abl).round(4).to_csv(config.RESULTS_DIR / "feature_ablation.csv", index=False)
    print(f"grid finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

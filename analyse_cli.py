"""Run the backend on one image, no UI. Prints the prediction, key features and timings,
and saves the intermediate images to an output folder.

  python analyse_cli.py assets/samples/melanoma.png
  python analyse_cli.py image.jpg --wavelet db4 --level 3 --no-hair --out results/cli_out
"""
import argparse
import json
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np

import config
from src import pipeline, preprocessing
from src.errors import InvalidImageError, NoLesionError

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    default = config.SAMPLES_DIR / "melanoma.png"
    ap.add_argument("image", nargs="?", default=str(default) if default.exists() else None,
                    help="image file (default: the melanoma demo sample, if generated)")
    ap.add_argument("--wavelet", default=config.WAVELET, choices=config.WAVELETS)
    ap.add_argument("--level", type=int, default=config.LEVEL, choices=config.LEVELS)
    ap.add_argument("--no-hair", action="store_true", help="skip hair removal")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "cli_out"))
    a = ap.parse_args()
    if a.image is None:
        ap.error("no image given (demo samples are created by make_figures.py)")

    try:
        path = Path(a.image)
        rgb = preprocessing.decode_image(path.read_bytes(), path.name)
        res = pipeline.analyse(rgb, a.wavelet, a.level, not a.no_hair,
                               model=joblib.load(config.MODEL_PATH))
    except (InvalidImageError, NoLesionError, FileNotFoundError) as e:
        raise SystemExit(f"ERROR: {type(e).__name__}: {e}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    im = res["images"]
    to_bgr = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / "hair_removed.png"), to_bgr(im["hair_removed"]))
    cv2.imwrite(str(out / "overlay.png"), to_bgr(im["overlay"]))
    cv2.imwrite(str(out / "final_mask.png"), im["final_mask"].astype(np.uint8) * 255)
    cv2.imwrite(str(out / "coarse_mask.png"), im["coarse_mask"].astype(np.uint8) * 255)

    p, m = res["prediction"], res["meta"]
    print(f"Prediction : {p['label']}  (melanoma score {p['score']:.2f}, flagged at >= {p['threshold']:.2f})")
    print(f"Settings   : {m['wavelet']} level {m['level']}, hair removal {'on' if m['hair_removal'] else 'off'}")
    if m["prediction_config_differs"]:
        mc = m["model_config"]
        print(f"Note       : prediction uses the default model ({mc['wavelet']}, level {mc['level']})")
    print(f"Time       : {m['time_ms']} ms  {m['timings']}")
    abcd = {k: round(v, 3) for k, v in res["features"].items() if not k.startswith(("L", "LL_"))}
    print("ABCD       :", json.dumps(abcd))
    print(f"Saved      : {out}/ (hair_removed, coarse_mask, final_mask, overlay)")


if __name__ == "__main__":
    main()

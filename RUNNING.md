# Running the project: start and stop

The project has two parts that you can run **separately**:

| Part | What it is | How it runs |
|---|---|---|
| **Backend** | `src/` pipeline + `train.py`, `experiments.py`, `analyse_cli.py` | Commands that run and exit. There is no backend server. |
| **Frontend** | `app.py` (Streamlit) | A web page served on your machine. It stays running until you stop it. |

The frontend calls the backend directly (`src.pipeline.analyse`), so the frontend needs the
backend's trained model file (`models/model.pkl`) but nothing else running.

All commands are run from the project folder (`skin_lesion_mra/`), on macOS or Linux.
On Windows replace `.venv/bin/python` with `.venv\Scripts\python`.

---

## 0. One-time setup

1. Open a terminal and go to the project folder:
   ```bash
   cd skin_lesion_mra
   ```
   (the folder that contains `app.py`, `config.py` and `requirements.txt`)
2. Create the virtual environment and install the libraries:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
3. Check the dataset is in place: `data/PH2/PH2 Dataset images/` (200 folders) and
   `data/PH2/PH2_dataset.txt`.
4. Check it all imports:
   ```bash
   .venv/bin/python -c "import cv2, pywt, sklearn, streamlit; print('ok')"
   ```
   You should see `ok`.

---

## 1. Backend

The backend has nothing to "stop": each command runs, prints its result and exits. If one is
taking too long, press **Ctrl+C**.

### 1a. Train the model (needed once, and again if you change `config.py`)
```bash
.venv/bin/python train.py
```
- Takes about 10 seconds.
- Prints the cross-validated metrics and ends with `saved .../models/model.pkl`.
- Writes `models/model.pkl`, `results/features_default.csv`, `results/cv_default.json`.

### 1b. Analyse one image from the command line (backend only, no UI)
```bash
.venv/bin/python analyse_cli.py assets/samples/melanoma.png
```
- Prints the prediction, melanoma score, ABCD features and timings.
- Saves the intermediate images to `results/cli_out/`.
- Options: `--wavelet haar|db4|sym4`, `--level 1|2|3`, `--no-hair`, `--out FOLDER`.
  ```bash
  .venv/bin/python analyse_cli.py my_image.jpg --wavelet db4 --level 3 --no-hair
  ```
- An unsupported or unreadable file prints `ERROR: ...` and exits with code 1.

### 1c. Run the experiment grid (optional, for the report)
```bash
.venv/bin/python experiments.py
```
- 18 configurations plus the feature ablation, about 1 minute.
- Writes `results/experiments.csv` and `results/feature_ablation.csv`.

### 1d. Regenerate report figures and the demo samples
(Needed once on a fresh clone if you want the sample buttons, because the PH2 sample images are not stored in the repository.)
```bash
.venv/bin/python make_figures.py
```
- Writes the figures to `results/figures/` and the three samples to `assets/samples/`.
- Run it after `train.py` and `experiments.py`.

---

## 2. Frontend

### 2a. Start
1. Make sure `models/model.pkl` exists (run step 1a if it does not).
2. Start the app:
   ```bash
   .venv/bin/streamlit run app.py
   ```
3. Your browser opens at **http://localhost:8501**. If it does not, open that address yourself.
   The terminal stays busy while the app is running. That is normal.
4. Use it:
   1. Click a sample (Common nevus, Atypical nevus or Melanoma) in the sidebar, or upload an image.
   2. Choose the wavelet, level and hair-removal setting.
   3. Click **Analyse**.
   4. Look through the **Pipeline**, **Result** and **About** tabs.

To use a different port: `.venv/bin/streamlit run app.py --server.port 8600`.

### 2b. Stop
- **Normal way:** click the terminal where it is running and press **Ctrl+C**. The terminal
  prompt comes back and the page stops responding.
- **If you lost the terminal** (or it is running in the background), find and stop it:
  ```bash
  lsof -ti :8501 | xargs kill
  ```
  Use your port number if you changed it. Check it stopped:
  ```bash
  lsof -i :8501
  ```
  This should print nothing.
- Closing the browser tab does **not** stop the app.

### 2c. Restart after changing code
Streamlit usually reloads when you save a file. If the page looks stale, press **R** in the
browser, or stop (2b) and start (2a) again. After changing `config.py` or any `src/` module that
affects features, run `train.py` again first, otherwise the model no longer matches.

---

## 3. Running both together (typical demo day)
1. `.venv/bin/python train.py` (only if `models/model.pkl` is missing or `config.py` changed).
2. `.venv/bin/streamlit run app.py`.
3. Demo with the three samples. Keep a screenshot and the backup video ready.
4. Press **Ctrl+C** in the terminal to stop.

---

## 4. Troubleshooting
| Problem | Fix |
|---|---|
| `FileNotFoundError: ... model.pkl` | Run `.venv/bin/python train.py` first. |
| `No module named 'cv2'` (or similar) | You are not using the virtual environment. Use `.venv/bin/python`, or re-run the install in step 0. |
| `Address already in use` / port busy | Stop the old app (step 2b), or use `--server.port 8600`. |
| Page opens but shows an error card | The uploaded file is not supported (JPG, PNG, BMP; at least 128×128; at most 10 MB), or no lesion was found. Try a clearer image. |
| `FileNotFoundError` for the dataset | Check `data/PH2/` as in step 0.3. Only `train.py`, `experiments.py` and `make_figures.py` need the dataset; the app and `analyse_cli.py` do not. |
| App shows the prediction but not for my settings | By design: the prediction always uses the default model (Haar, level 2). Your settings change the displayed steps only. |

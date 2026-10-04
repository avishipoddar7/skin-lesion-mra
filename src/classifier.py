"""Stage 5: StandardScaler + calibrated RBF SVM."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config

LABELS = {0: "Non-melanoma", 1: "Melanoma"}


def make_model():
    svm = SVC(kernel="rbf", C=config.SVM_C, gamma="scale", class_weight="balanced",
              random_state=config.SEED)
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(svm, method="sigmoid", cv=config.CV_FOLDS, ensemble=False))


def train(X, y):
    return make_model().fit(X, y)


def melanoma_score(model, X):
    """Estimated probability of melanoma for each row of X."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def predict(model, x, threshold=config.DECISION_THRESHOLD):
    """x: 1-D feature vector. Returns (label, melanoma_score). The score is a model
    estimate, not a clinically calibrated probability."""
    score = float(melanoma_score(model, x)[0])
    return LABELS[int(score >= threshold)], score

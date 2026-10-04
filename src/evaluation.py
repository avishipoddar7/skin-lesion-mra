"""Cross-validated classification metrics."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
from src import classifier, metrics


def cv_evaluate(X, y, seeds=(42, 43, 44), threshold=config.DECISION_THRESHOLD):
    """Stratified 5-fold CV repeated over `seeds`. Returns mean/std of each metric plus
    the pooled out-of-fold scores of the first seed (for ROC and confusion matrix)."""
    X, y = np.asarray(X, float), np.asarray(y)
    runs, first = [], None
    for seed in seeds:
        cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed)
        scores = cross_val_predict(classifier.make_model(), X, y, cv=cv,
                                   method="predict_proba")[:, 1]
        pred = (scores >= threshold).astype(int)
        sens, spec = metrics.sensitivity_specificity(y, pred)
        runs.append({"accuracy": float((pred == y).mean()), "sensitivity": sens,
                     "specificity": spec, "auc": float(roc_auc_score(y, scores))})
        if first is None:
            first = {"scores": scores, "pred": pred,
                     "confusion": confusion_matrix(y, pred, labels=[0, 1])}
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs])
        out[k] = float(v.mean())
        out[k + "_std"] = float(v.std())
    out["first"] = first
    return out

"""Train abuse-probability model + reason classifier with calibration.
Outputs: models/returnwise_model.joblib, models/thresholds.json, models/metrics.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from features import build_preprocessor, text_meta, META_COLS, PlattCalibrated
from sklearn.metrics import average_precision_score, brier_score_loss
import joblib

from features import build_preprocessor, text_meta, META_COLS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

# Cost matrix (expected-cost routing). Defended ratio: 1 wrong block (genuine->escalate) ~= 10 missed frauds (abusive->auto).
# C[true][pred]; pred order: auto-approve, inspect, escalate
COST = {
    "abusive": {"auto-approve": 1.0, "inspect": 0.3, "escalate": 0.0},
    "genuine": {"auto-approve": 0.0, "inspect": 0.5, "escalate": 10.0},
}
# genuine_* labels share the "genuine" cost row
LABELS_GENUINE = {"genuine_quality_issue", "genuine_fit_issue", "genuine_wrong_item"}


def tune_thresholds(p_valid: np.ndarray, y_abuse_valid: np.ndarray):
    """Grid-search (t_esc, t_insp) thresholds minimising total cost on validation.
    Decision: p>=t_esc -> escalate; p>=t_insp -> inspect; else auto-approve."""
    best = None
    for t_esc in np.arange(0.4, 0.95, 0.05):
        for t_insp in np.arange(0.1, 0.6, 0.05):
            if t_insp >= t_esc:
                continue
            pred = np.where(p_valid >= t_esc, "escalate",
                            np.where(p_valid >= t_insp, "inspect", "auto-approve"))
            cost = 0.0
            for pr, yt in zip(pred, y_abuse_valid):
                row = "abusive" if yt else "genuine"
                cost += COST[row][pr]
            avg = cost / len(pred)
            if best is None or avg < best[0]:
                best = (avg, float(t_esc), float(t_insp))
    return {"avg_cost": best[0], "t_escalate": best[1], "t_inspect": best[2]}


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "returns.csv")
    meta = text_meta(df)
    for c in META_COLS:
        df[c] = meta[c].values

    feature_cols = ["customer_reason"]  # preprocessor pulls the rest by name
    X_train = df[df.split == "train"].reset_index(drop=True)
    X_valid = df[df.split == "valid"].reset_index(drop=True)
    y_abuse_train = (X_train["label"] == "abusive").astype(int).values
    y_abuse_valid = (X_valid["label"] == "abusive").astype(int).values
    y_reason_train = X_train["label"].values

    pre = build_preprocessor()
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         max_leaf_nodes=63, min_samples_leaf=50,
                                         l2_regularization=5.0, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", base)])
    pipe.fit(X_train, y_abuse_train)
    # Platt (sigmoid) calibration on validation split
    cal = PlattCalibrated(pipe).fit(X_valid, y_abuse_valid)

    p_valid = cal.predict_proba(X_valid)[:, 1]
    print(f"valid AP={average_precision_score(y_abuse_valid, p_valid):.3f} "
          f"Brier={brier_score_loss(y_abuse_valid, p_valid):.4f}")

    thr = tune_thresholds(p_valid, y_abuse_valid)
    print("tuned thresholds:", thr)

    # reason-category classifier (for justification + resolution tone), trained on genuine+abusive
    pre2 = build_preprocessor()
    reason_clf = Pipeline([("pre", pre2),
                           ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                                                 max_leaf_nodes=63, min_samples_leaf=50,
                                                                 l2_regularization=5.0, random_state=7))])
    reason_clf.fit(X_train, y_reason_train)

    joblib.dump({"abuse_model": cal, "reason_model": reason_clf, "cost": COST}, MODELS / "returnwise_model.joblib")
    json.dump(thr, open(MODELS / "thresholds.json", "w"), indent=2)
    # store validation metrics
    json.dump({"valid_AP": float(average_precision_score(y_abuse_valid, p_valid)),
               "valid_Brier": float(brier_score_loss(y_abuse_valid, p_valid))},
              open(MODELS / "metrics.json", "w"), indent=2)
    print("saved models/")


if __name__ == "__main__":
    main()

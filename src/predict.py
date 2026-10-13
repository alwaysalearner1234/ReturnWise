"""Inference: calibrated P(abuse) -> 3-way routing + justification + resolution message."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from features import text_meta, META_COLS

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

_DECISIONS = ["auto-approve", "inspect", "escalate"]

RESOLUTIONS = {
    "genuine_quality_issue": "We're sorry your item arrived with a quality problem — your return is {route}. A prepaid pickup will be arranged, and we'll refund/replace you on priority after a quick quality check.",
    "genuine_fit_issue": "Thanks for flagging the fit issue — your return is {route}. You may pick a different size for instant exchange, and we've noted the size feedback to fix our size chart.",
    "genuine_wrong_item": "Sorry we sent the wrong item — your return is {route}. We'll arrange a doorstep swap for the correct item plus a small goodwill coupon.",
    "abusive": "Thanks — your request is with our specialist {route} team. To keep returns fair for everyone, they will verify the item condition and update you within 48 hours.",
}


def load_artifacts():
    bundle = joblib.load(MODELS / "returnwise_model.joblib")
    thr = json.load(open(MODELS / "thresholds.json"))
    return bundle["abuse_model"], bundle["reason_model"], thr


def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    meta = text_meta(df)
    for c in META_COLS:
        if c not in df.columns:
            df[c] = meta[c].values
    return df


def route(p: np.ndarray, thr: dict) -> np.ndarray:
    return np.where(p >= thr["t_escalate"], "escalate",
                    np.where(p >= thr["t_inspect"], "inspect", "auto-approve"))


def justify(row: pd.Series, p_abuse: float, decision: str, reason_cat: str) -> str:
    bits = []
    if row.get("returns_prev_60d", 0) >= 3:
        bits.append(f"{int(row['returns_prev_60d'])} returns in last 60 days")
    if row.get("prior_return_rate", 0) >= 0.5:
        bits.append(f"prior return rate {float(row['prior_return_rate']):.0%}")
    if row.get("same_size_returns_60d", 0) >= 2:
        bits.append(f"{int(row['same_size_returns_60d'])} returns of same size in 60 days")
    if row.get("days_since_delivery", 0) >= 22:
        bits.append(f"return filed {int(row['days_since_delivery'])} days after delivery (near 30-day limit)")
    if row.get("discount_used", 0) >= 0.3 and row.get("order_value", 0) >= 1500:
        bits.append(f"high discount ({float(row['discount_used']):.0%}) on high-value order (Rs {float(row['order_value']):.0f})")
    if row.get("payment_method") == "COD" and row.get("prior_return_rate", 0) >= 0.4:
        bits.append("COD order with elevated return history")
    txt = str(row.get("customer_reason", ""))
    low = txt.lower()
    if any(k in low for k in ["party", "shaadi", "pehen", "wore", "trial", "function"]):
        bits.append("reason text suggests wardrobing/use-then-return")
    if any(k in low for k in ["khali", "empty", "missing", "gayab"]):
        bits.append("reason claims empty/missing box (unverifiable without images)")
    if any(k in low for k in ["tight", "loose", "chhota", "bada", "size", "fit", "dheela"]):
        bits.append("reason text describes fit/size mismatch")
    if any(k in low for k in ["tear", "phata", "stain", "dag", "defect", "silai", "hole", "ched", "fade", "sole"]):
        bits.append("reason text describes quality defect")
    if any(k in low for k in ["wrong", "galat", "different", "alag"]):
        bits.append("reason text claims wrong item delivered")
    if not bits:
        bits.append("no strong abuse pattern; profile consistent with one-off genuine return")
    conf = f"P(abuse)={p_abuse:.2f}"
    return f"[{decision} | {conf} | text-category={reason_cat}] " + "; ".join(bits)


def resolution(reason_cat: str, decision: str) -> str:
    route_word = {"auto-approve": "approved", "inspect": "marked for a quick quality inspection",
                  "escalate": "review"}.get(decision, decision)
    return RESOLUTIONS.get(reason_cat, RESOLUTIONS["genuine_quality_issue"]).format(route=route_word)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    abuse_model, reason_model, thr = load_artifacts()
    df = add_meta(df)
    p = abuse_model.predict_proba(df)[:, 1]
    decisions = route(p, thr)
    reason_cats = reason_model.predict(df)
    out = df.copy()
    out["p_abuse"] = np.round(p, 4)
    out["decision"] = decisions
    out["reason_category"] = reason_cats
    out["justification"] = [justify(r, pp, d, rc) for r, pp, d, rc in
                            zip(df.to_dict("records"), p, decisions, reason_cats)]
    out["resolution_message"] = [resolution(rc, d) for rc, d in zip(reason_cats, decisions)]
    return out[["request_id", "p_abuse", "decision", "reason_category", "justification", "resolution_message"]]

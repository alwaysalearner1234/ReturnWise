"""Shared feature engineering: structured + text (no images).
IMPORTANT: reason_lang is EXCLUDED from features (fairness: language must not drive flags).
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as _np


class PlattCalibrated(BaseEstimator, ClassifierMixin):
    """Prefit pipe + sigmoid (Platt) scaling fit on a held-out split."""
    def __init__(self, pipe=None):
        self.pipe = pipe
    def fit(self, X, y):
        s = self.pipe.predict_proba(X)[:, 1].reshape(-1, 1)
        self.scaler_ = LogisticRegression().fit(s, y)
        self.classes_ = _np.array([0, 1])
        return self
    def predict_proba(self, X):
        s = self.pipe.predict_proba(X)[:, 1].reshape(-1, 1)
        p = self.scaler_.predict_proba(s)[:, 1]
        return _np.stack([1 - p, p], axis=1)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

STRUCT_NUM = [
    "order_value", "discount_used", "delivery_time_days", "days_since_delivery",
    "customer_tenure_days", "prior_return_rate", "returns_prev_60d",
    "same_size_returns_60d",
]
STRUCT_CAT = ["category", "payment_method", "size_ordered"]

KEYWORDS_QUALITY = ["tear", "torn", "phata", "stain", "dag", "defect", "silai", "zip", "button", "sole", "hole", "ched", "fade", "peel", "thread", "dhaag", "kharab"]
KEYWORDS_FIT = ["tight", "loose", "size", "saij", "fit", "chhota", "bada", "dheela", "length", "oversiz", "small", "big", "kamar"]
KEYWORDS_WRONG = ["wrong", "galat", "different", "alag", "mix", "instead", "dusra", "exchange"]
KEYWORDS_ABUSE = ["party", "shaadi", "pehen", "wore", "trial", "keeping one", "cheaper elsewhere", "no reason", "pasand nahi", "function", "khali", "empty", "missing"]


def _clean(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_meta(df: pd.DataFrame) -> pd.DataFrame:
    raw = df["customer_reason"].fillna("").astype(str)
    clean = raw.apply(_clean)
    out = pd.DataFrame(index=df.index)
    out["txt_len"] = raw.str.len()
    out["txt_words"] = raw.str.split().str.len().fillna(0)
    out["caps_ratio"] = raw.apply(lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1))
    out["exclaim"] = raw.str.count("!") + raw.str.count(r"\.\.\.")
    out["digit"] = raw.str.count(r"\d")
    for name, kws in [("kw_quality", KEYWORDS_QUALITY), ("kw_fit", KEYWORDS_FIT),
                      ("kw_wrong", KEYWORDS_WRONG), ("kw_abuse", KEYWORDS_ABUSE)]:
        out[name] = clean.apply(lambda s, k=kws: sum(1 for w in k if w in s))
    # derived risk interactions (structured-only, explainable)
    out["serial_returner"] = ((df["returns_prev_60d"] >= 3) | (df["prior_return_rate"] >= 0.5)).astype(int)
    out["repeat_size_pattern"] = (df["same_size_returns_60d"] >= 2).astype(int)
    out["late_return"] = (df["days_since_delivery"] >= 22).astype(int)
    out["high_discount_high_value"] = ((df["discount_used"] >= 0.3) & (df["order_value"] >= 1500)).astype(int)
    out["new_customer"] = (df["customer_tenure_days"] <= 30).astype(int)
    out["low_value"] = (df["order_value"] <= 600).astype(int)
    return out


META_COLS = ["txt_len", "txt_words", "caps_ratio", "exclaim", "digit",
             "kw_quality", "kw_fit", "kw_wrong", "kw_abuse",
             "serial_returner", "repeat_size_pattern", "late_return",
             "high_discount_high_value", "new_customer", "low_value"]


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))])
    meta_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=_clean, analyzer="word",
                                  ngram_range=(1, 2), max_features=4000,
                                  sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=96, random_state=42)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, STRUCT_NUM),
        ("cat", cat_pipe, STRUCT_CAT),
        ("meta", meta_pipe, META_COLS),
        ("txt", text_pipe, "customer_reason"),
    ])
    return pre

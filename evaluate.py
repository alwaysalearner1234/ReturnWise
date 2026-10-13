"""evaluate.py — cost-weighted scoring + fairness check + size-chart insight.
Usage: python evaluate.py [--split test] [--out predictions.csv]
Score = cost-weighted accuracy using the published cost matrix, NOT raw accuracy.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from predict import predict  # noqa: E402

# Published cost matrix C[true_is_abuse][pred]. Defended ratio 10:1:
# one wrongful escalation of a genuine customer (10.0) ~= ten missed frauds (1.0 each).
COST_ABUSIVE = {"auto-approve": 1.0, "inspect": 0.3, "escalate": 0.0}
COST_GENUINE = {"auto-approve": 0.0, "inspect": 0.5, "escalate": 10.0}
MAX_COST = 10.0


def row_cost(is_abuse: bool, pred: str) -> float:
    return (COST_ABUSIVE if is_abuse else COST_GENUINE)[pred]


def fairness_report(test: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    df = test.merge(preds[["request_id", "decision"]], on="request_id")
    df["flagged"] = (df["decision"] == "escalate").astype(int)
    df["tenure_grp"] = pd.cut(df["customer_tenure_days"], [-1, 30, 180, 10**9],
                              labels=["new<=30d", "mid31-180d", "old>180d"])
    df["value_grp"] = pd.cut(df["order_value"], [-1, 600, 1500, 10**9],
                             labels=["low<=600", "mid601-1500", "high>1500"])
    df["lang_grp"] = df["reason_lang"]
    rows = []
    for col in ["tenure_grp", "value_grp", "lang_grp"]:
        g = df.groupby(col, observed=True)["flagged"].agg(["mean", "count"]).reset_index()
        g["dimension"] = col
        g.rename(columns={"mean": "escalate_rate", "count": "n"}, inplace=True)
        overall = df["flagged"].mean()
        g["overall_rate"] = overall
        g["disparity_ratio"] = g["escalate_rate"] / max(overall, 1e-9)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def sizechart_fix(test: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    df = test.merge(preds[["request_id", "reason_category"]], on="request_id")
    fit = df[df["reason_category"] == "genuine_fit_issue"]
    g = fit.groupby("product_id").agg(fit_returns=("product_id", "size"),
                                      total_returns=("request_id", "size")).reset_index()
    # join catalogue totals from full test for rate
    tot = df.groupby("product_id").size().rename("t").reset_index()
    g = g.merge(tot, on="product_id", how="left")
    g["fit_rate"] = g["fit_returns"] / g["t"]
    return g.sort_values(["fit_returns", "fit_rate"], ascending=False).head(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    test = pd.read_csv(ROOT / "data" / "returns.csv")
    test = test[test["split"] == args.split].reset_index(drop=True)
    preds = predict(test)

    is_abuse = (test.set_index("request_id").loc[preds["request_id"], "label"] == "abusive").values
    costs = np.array([row_cost(a, p) for a, p in zip(is_abuse, preds["decision"].values)])
    avg_cost = costs.mean()
    cost_weighted_acc = 1 - avg_cost / MAX_COST
    raw_acc = (((preds["decision"] == "escalate") == is_abuse) |
               ((preds["decision"] != "escalate") & (~is_abuse))).mean()
    # confusion: abusive recall on escalate, genuine wrongly-escalated rate
    abuse_rec = ((preds["decision"] == "escalate") & is_abuse).sum() / max(is_abuse.sum(), 1)
    false_block = ((preds["decision"] == "escalate") & (~is_abuse)).sum() / max((~is_abuse).sum(), 1)

    print("=== COST MATRIX (published) ===")
    print(f"  abusive -> auto:{COST_ABUSIVE['auto-approve']} inspect:{COST_ABUSIVE['inspect']} escalate:{COST_ABUSIVE['escalate']}")
    print(f"  genuine -> auto:{COST_GENUINE['auto-approve']} inspect:{COST_GENUINE['inspect']} escalate:{COST_GENUINE['escalate']}")
    print(f"  Defended ratio: 1 wrongful escalate (10.0) ~= 10 missed frauds (10x1.0).")
    print(f"\nN={len(test)}  total_cost={costs.sum():.1f}  avg_cost={avg_cost:.4f}")
    print(f"COST-WEIGHTED ACCURACY (official score) = {cost_weighted_acc:.4f}")
    print(f"raw accuracy (reference only)           = {raw_acc:.4f}")
    print(f"abusive escalate-recall                 = {abuse_rec:.4f}")
    print(f"genuine wrong-escalate rate             = {false_block:.4f}")
    print("\ndecision mix:\n", preds["decision"].value_counts(normalize=True).round(3).to_string())

    fair = fairness_report(test, preds)
    print("\n=== FAIRNESS CHECK (escalate rate by group; PASS if max disparity ratio < 2.0 and new/low/non-English not over-flagged) ===")
    print(fair.to_string(index=False))
    worst = fair["disparity_ratio"].max()
    print(f"max disparity ratio = {worst:.2f} -> {'PASS' if worst < 2.0 else 'FAIL'}")

    fix = sizechart_fix(test, preds)
    print("\n=== SIZE-CHART FIX: top-5 fit-return products ===")
    print(fix.to_string(index=False))

    preds_out = test[["request_id"]].merge(preds, on="request_id")
    preds_out.to_csv(ROOT / args.out, index=False)
    fair.to_csv(ROOT / "fairness_report.csv", index=False)
    fix.to_csv(ROOT / "sizechart_fix.csv", index=False)
    print(f"\nwrote {args.out}, fairness_report.csv, sizechart_fix.csv")


if __name__ == "__main__":
    main()

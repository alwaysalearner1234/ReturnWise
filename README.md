# R2 — Returns Fraud vs. Genuine Grievance Sorter (ReturnWise)

Three-way return routing (**auto-approve / inspect / escalate**) with calibrated
confidence, human-readable justifications, cost-weighted scoring, fairness audit,
and size-chart insights. Multilingual messy text (EN / Hinglish / transliterated Hindi).

## Cost model (defended 10:1)

False accusation destroys loyalty; a missed fraud only costs one refund.

| true \ pred | auto-approve | inspect | escalate |
|---|---|---|---|
| abusive | 1.0 (missed fraud) | 0.3 | 0.0 |
| genuine_* | 0.0 | 0.5 (friction) | **10.0 (wrongful block)** |

So **1 wrongful block ≈ 10 missed frauds**. Thresholds `(t_inspect, t_escalate)`
are grid-tuned on the validation split to minimise total cost, and inference uses
`P(abuse)` from a calibrated `HistGradientBoosting + sigmoid calibration` model.
Official score: `cost_weighted_accuracy = 1 − avg_cost/10`.

`reason_lang` is **excluded from features** — language must never drive flags.

## Layout

```
data/returns.csv      60k synthetic requests (train/valid/test) + reason_lang + hidden label
data/catalogue.csv    product catalogue (5 known bad-fit products, v1-outdated charts)
src/generate_data.py  regenerates data (seed 42)
src/features.py       structured + TF-IDF + keyword/meta features
src/train.py          trains abuse + reason models, tunes cost thresholds
src/predict.py        routing + justification + tone-matched resolution message
evaluate.py           official scorer: cost-weighted acc, fairness, size-chart fix
```

## Quickstart

```powershell
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
python evaluate.py --split test
```

## Outputs per decision

- `p_abuse`: calibrated confidence
- `decision`: auto-approve / inspect / escalate
- `justification`: cites exact signals, e.g.
  `"[escalate | P(abuse)=0.87] 4 returns in last 60 days; 3 returns of same size in 60 days; reason text suggests wardrobing/use-then-return"`
- `resolution_message`: tone-matched to predicted reason category
  (quality → apology+priority refund; fit → exchange+nudge; wrong-item → doorstep swap; abusive → polite 48h specialist review, never an accusation)
```

## Fairness (scored)

`evaluate.py` prints escalate rates by `tenure_grp × value_grp × reason_lang` with
disparity ratios vs overall. PASS if max ratio < 2.0. Mitigations: language-blind
features, group-blind thresholds, conservative escalate threshold from the 10:1 cost ratio.

## Size-chart fix

Top-5 products by fit-related return count/rate → correct their size guidance first.
See `sizechart_fix.csv`.

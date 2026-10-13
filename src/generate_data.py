"""Synthetic data generator for R2: Returns Fraud vs Genuine Grievance Sorter.
Generates 60k return requests + product catalogue with hidden labels.
Deterministic via seed. No external deps beyond pandas/numpy.
"""
from __future__ import annotations
import random
import string
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 60000
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORIES = ["kurta", "saree", "dress", "jeans", "tshirt", "shirt", "lehenga", "jacket", "shoes", "handbag"]
PAYMENTS = ["COD", "UPI", "card", "netbanking", "wallet"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL"]

# 5 products with broken size guidance (fit-return traps)
BAD_FIT_PRODUCTS = ["P-KUR-014", "P-JEA-033", "P-DRE-007", "P-SHO-021", "P-SHI-002"]

REASON_POOLS = {
    "genuine_quality_issue": {
        "en": [
            "Stitching came undone on first wash, threads hanging near the zipper",
            "Fabric has a tear near the shoulder seam, poor quality",
            "Colour faded badly after one wash, looks old already",
            "Button missing and there is a stain on the front panel",
            "Sole of the shoe is coming off, glue visible, defective piece",
            "Received with a hole in the lining and loose threads everywhere",
            "Print is peeling off and cloth feels very cheap",
        ],
        "hinglish": [
            "Silai khul gayi first wash me hi, zip ke paas dhaage nikle hue hain",
            "Kapde me shoulder ke paas tear hai, quality bahut kharab hai",
            "Ek wash me hi colour utar gaya, purana lag raha hai",
            "Button missing hai aur front par stain bhi hai",
            "Shoe ka sole nikal raha hai, defective piece mila hai",
        ],
        "translit": [
            "kapda phata hua aya, silai ukhdi hui hai",
            "dag laga hua kurta mila, EXCHANGE karo",
            "jutte me chipka hua sole, tuta hua aya",
            "sadi me ched hai, kharab quality bheja",
        ],
    },
    "genuine_fit_issue": {
        "en": [
            "Size runs too small, M feels like S, very tight around chest",
            "Too loose fitting, size chart said M but it is oversized",
            "Length is too short, size issue, need bigger size",
            "Ordered L but fits like XL, please exchange for correct fit",
            "Waist is too tight, size chart measurements seem wrong",
        ],
        "hinglish": [
            "Size chhota aya, M manga tha S jaisa lag raha hai, tight hai",
            "Bahut loose hai, size chart me M likha tha par bada aya",
            "Length chhoti hai, bada size chahiye, fitting issue hai",
            "L order kiya tha XL jaisa aya, fit sahi nahi hai",
            "Kamar par tight ho raha hai, size chart galat lagta hai",
        ],
        "translit": [
            "saij chota aya, badalna hai",
            "kurta bahut dheela aya, size chart theek karo",
            "pant kamar par tight, bada size do",
        ],
    },
    "genuine_wrong_item": {
        "en": [
            "Received wrong colour, ordered red got blue instead",
            "Wrong size delivered, box says M but item inside is L",
            "Got a completely different product, not what I ordered",
            "Ordered kurta received saree, packing mix-up please replace",
            "Size and design both different from the pictures",
        ],
        "hinglish": [
            "Galat colour aya, red manga tha blue mil gaya",
            "Galat size deliver hua, dabbe par M likha andar L nikla",
            "Bilkul alag product mil gaya, jo order kiya tha vo nahi hai",
            "Kurta order kiya saree aa gayi, please replace karo",
        ],
        "translit": [
            "galat saman aya, jo manga vo nahi bheja",
            "rang galat bheja, neela manga lal aya",
            "dabba dusra aya, item change karke do",
        ],
    },
    "abusive": {
        "en": [
            "Did not like it, just return it, no reason needed",
            "Wore it for a party, now returning, tag removed but ok",
            "Quality is fine but I found it cheaper elsewhere so returning",
            "Ordered 3 sizes to try at home, keeping one, returning two",
            "Box is empty, item missing, refund me fast",  # possible empty-box claim
            "Not needed anymore, function got cancelled, take it back",
        ],
        "hinglish": [
            "Shaadi me ek baar pehen liya, ab return kar rahi hu, koi issue nahi hai",
            "3 size order kiye the trial ke liye, ek rakhungi do wapas",
            "Pasand nahi aya bas, return kar do, koi reason nahi",
            "Party me pehen liya ab wapas le lo, thoda stain ho gaya to kya",
            "Khali dabba aya hai, saman gayab, refund do jaldi",
        ],
        "translit": [
            "ek bar pehen liya ab vapas karna hai",
            "teen mangaye ek rakhna do vapas",
            "pasand nahi aya vapas le lo bas",
            "khali dibba mila refund chahiye",
        ],
    },
}

TYPO_FN = [
    lambda w: w[:-1] if len(w) > 4 and random.random() < 0.5 else w,
    lambda w: w + w[-1] if len(w) > 2 else w,
    lambda w: w.replace("a", "aa", 1) if "a" in w else w,
    lambda w: w.replace("i", "e", 1) if "i" in w else w,
]


def add_typos(text: str, p: float = 0.06) -> str:
    out = []
    for w in text.split():
        if random.random() < p and len(w) > 3:
            w = random.choice(TYPO_FN)(w)
            # random case noise
            if random.random() < 0.3:
                w = w.upper() if random.random() < 0.5 else w.lower()
        out.append(w)
    # occasional missing space / extra punctuation
    s = " ".join(out)
    if random.random() < 0.08:
        s = s.replace(" ", "", 1)
    if random.random() < 0.1:
        s += random.choice(["!!", "...", " plz", " pls refund"])
    return s


def pick_reason(label: str, rng: np.random.Generator) -> tuple[str, str]:
    r = rng.random()
    if r < 0.60:
        lang = "en"
    elif r < 0.85:
        lang = "hinglish"
    else:
        lang = "translit"
    pool = REASON_POOLS[label][lang]
    text = random.choice(pool)
    text = add_typos(text)
    return text, lang


def main(n: int = N, seed: int = SEED):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- catalogue ----
    n_products = 120
    prod_ids = [f"P-{c[:3].upper()}-{i:03d}" for i, c in enumerate(rng.choice(CATEGORIES, n_products))]
    # ensure bad-fit products exist with right categories
    catalogue = pd.DataFrame({
        "product_id": [f"P-PRO-{i:03d}" for i in range(n_products)],
        "category": rng.choice(CATEGORIES, n_products),
        "brand_tier": rng.choice(["budget", "mid", "premium"], n_products, p=[0.5, 0.35, 0.15]),
        "price_base": rng.integers(299, 4999, n_products),
        "size_chart_version": rng.choice(["v1", "v2", "v3"], n_products),
    })
    # force 5 bad-fit products in
    for i, pid in enumerate(BAD_FIT_PRODUCTS):
        catalogue.loc[i, "product_id"] = pid
        catalogue.loc[i, "size_chart_version"] = "v1-outdated"
    catalogue.to_csv(OUT_DIR / "catalogue.csv", index=False)

    # ---- customers ----
    n_customers = 15000
    cust_ids = [f"C{i:06d}" for i in range(n_customers)]
    tenure = rng.integers(5, 1200, n_customers)  # days
    base_return_rate = rng.beta(2, 8, n_customers)  # mostly low
    # a small abusive subpopulation (~6%) with high return rates
    is_abuser_persona = rng.random(n_customers) < 0.06
    base_return_rate[is_abuser_persona] = rng.beta(8, 2, is_abuser_persona.sum()) * 0.9 + 0.1
    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "tenure_days": tenure,
        "persona_abuser": is_abuser_persona.astype(int),
        "base_return_rate": base_return_rate,
    })

    # ---- requests ----
    rows = []
    # track per-customer reason reuse & counts for fraud realism
    cust_reason_memory: dict[str, list[str]] = {}
    cust_return_count_60d = {}

    for i in range(n):
        ci = rng.integers(0, n_customers)
        cust = customers.iloc[ci]
        prod = catalogue.iloc[rng.integers(0, n_products)]
        category = prod["category"]
        product_id = prod["product_id"]

        # label: ~12% abusive overall
        if cust["persona_abuser"] and rng.random() < 0.75:
            label = "abusive"
        else:
            r = rng.random()
            if r < 0.12:
                label = "abusive"
            elif r < 0.42:
                label = "genuine_fit_issue"
            elif r < 0.70:
                label = "genuine_quality_issue"
            else:
                label = "genuine_wrong_item"

        # structured fields conditioned on label
        if label == "abusive":
            order_value = float(rng.choice([rng.integers(1500, 5000), rng.integers(299, 1200)], p=[0.55, 0.45]))
            discount = float(rng.choice([rng.uniform(0.3, 0.6), rng.uniform(0.0, 0.2)], p=[0.5, 0.5]))
            prior_return_rate = float(np.clip(cust["base_return_rate"] + rng.uniform(0, 0.25), 0, 1))
            n_prev_60d = int(rng.choice([2, 3, 4, 5, 6], p=[0.2, 0.25, 0.25, 0.15, 0.15]))
            same_size_60d = int(rng.choice([2, 3, 4], p=[0.4, 0.35, 0.25]))
            days_since_delivery = int(rng.choice([22, 25, 27, 29, 10, 15], p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1]))
            delivery_days = int(rng.integers(2, 6))
            payment = str(rng.choice(PAYMENTS, p=[0.35, 0.25, 0.25, 0.1, 0.05]))  # COD-heavy
            size = str(rng.choice(SIZES))
            # stealth abusers: ~20% keep a low profile (few traces in structured history)
            if rng.random() < 0.20:
                prior_return_rate = float(np.clip(rng.beta(2, 8), 0, 0.4))
                n_prev_60d = int(rng.choice([0, 1], p=[0.6, 0.4]))
                same_size_60d = int(rng.choice([0, 1], p=[0.8, 0.2]))
                days_since_delivery = int(rng.integers(3, 18))
        elif label == "genuine_fit_issue":
            # biased to bad-fit products
            if rng.random() < 0.35:
                prod = catalogue[catalogue.product_id.isin(BAD_FIT_PRODUCTS)].sample(1, random_state=rng).iloc[0]
                product_id = prod["product_id"]
                category = prod["category"]
            order_value = float(rng.integers(399, 2999))
            discount = float(rng.uniform(0.0, 0.35))
            prior_return_rate = float(np.clip(cust["base_return_rate"] * 0.6, 0, 1))
            n_prev_60d = int(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
            same_size_60d = int(rng.choice([0, 1, 2], p=[0.55, 0.3, 0.15]))
            days_since_delivery = int(rng.integers(2, 15))
            delivery_days = int(rng.integers(2, 9))
            payment = str(rng.choice(PAYMENTS, p=[0.2, 0.3, 0.3, 0.1, 0.1]))
            size = str(rng.choice(["M", "L", "S", "XL"]))
            # heavy-but-honest returners: ~8% of genuine fit buyers also return often
            if rng.random() < 0.08:
                prior_return_rate = float(np.clip(rng.uniform(0.4, 0.7), 0, 1))
                n_prev_60d = int(rng.choice([3, 4], p=[0.6, 0.4]))
                same_size_60d = int(rng.choice([2, 3], p=[0.6, 0.4]))
        elif label == "genuine_quality_issue":
            order_value = float(rng.integers(299, 2499))
            discount = float(rng.uniform(0.0, 0.3))
            prior_return_rate = float(np.clip(cust["base_return_rate"] * 0.5, 0, 1))
            n_prev_60d = int(rng.choice([0, 1], p=[0.8, 0.2]))
            same_size_60d = int(rng.choice([0, 1], p=[0.85, 0.15]))
            days_since_delivery = int(rng.integers(1, 12))
            delivery_days = int(rng.integers(2, 9))
            payment = str(rng.choice(PAYMENTS, p=[0.2, 0.3, 0.3, 0.1, 0.1]))
            size = str(rng.choice(SIZES))
        else:  # genuine_wrong_item
            order_value = float(rng.integers(399, 2999))
            discount = float(rng.uniform(0.0, 0.25))
            prior_return_rate = float(np.clip(cust["base_return_rate"] * 0.5, 0, 1))
            n_prev_60d = int(rng.choice([0, 1], p=[0.85, 0.15]))
            same_size_60d = 0
            days_since_delivery = int(rng.integers(1, 10))
            delivery_days = int(rng.choice([rng.integers(7, 14), rng.integers(2, 6)], p=[0.4, 0.6]))
            payment = str(rng.choice(PAYMENTS, p=[0.2, 0.3, 0.3, 0.1, 0.1]))
            size = str(rng.choice(SIZES))

        # reason text: abusers sometimes copy-paste prior reasons; ~30% fabricate a
        # genuine-sounding excuse (text alone is NOT sufficient evidence by design)
        mem = cust_reason_memory.setdefault(cust["customer_id"], [])
        if label == "abusive" and rng.random() < 0.30:
            fab_label = str(rng.choice(["genuine_quality_issue", "genuine_fit_issue", "genuine_wrong_item"]))
            text, lang = pick_reason(fab_label, rng)
            mem.append(text)
            if len(mem) > 5:
                mem.pop(0)
        elif label == "abusive" and mem and rng.random() < 0.35:
            text = random.choice(mem)  # reuse -> detectable signal
            # detect lang crudely
            lang = "en" if all(ord(c) < 128 for c in text) and "hai" not in text.lower().split() else "hinglish"
        else:
            text, lang = pick_reason(label, rng)
            mem.append(text)
            if len(mem) > 5:
                mem.pop(0)

        rows.append({
            "request_id": f"R{i:06d}",
            "customer_id": cust["customer_id"],
            "product_id": product_id,
            "category": category,
            "size_ordered": size,
            "order_value": round(order_value, 2),
            "discount_used": round(discount, 3),
            "delivery_time_days": delivery_days,
            "days_since_delivery": days_since_delivery,
            "customer_tenure_days": int(cust["tenure_days"]),
            "prior_return_rate": round(prior_return_rate, 3),
            "returns_prev_60d": n_prev_60d,
            "same_size_returns_60d": same_size_60d,
            "payment_method": payment,
            "customer_reason": text,
            "reason_lang": lang,  # for fairness audit only; NOT a model feature
            "label": label,
        })

    df = pd.DataFrame(rows)
    # time-based split: first 70% train, next 15% valid, last 15% test (by request order)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)
    df["split"] = "test"
    df.loc[df.index[:n_train], "split"] = "train"
    df.loc[df.index[n_train:n_train + n_valid], "split"] = "valid"
    df.to_csv(OUT_DIR / "returns.csv", index=False)
    print(f"Wrote {len(df)} rows to {OUT_DIR / 'returns.csv'}")
    print(df["label"].value_counts(normalize=True).round(3))
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()

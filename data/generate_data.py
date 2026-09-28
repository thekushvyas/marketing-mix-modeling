"""
Generate a realistic *synthetic* weekly marketing dataset for a direct-to-consumer brand.

Why synthetic? Real spend/sales data is almost always confidential. Simulating it with
known "ground-truth" parameters lets us check whether the model actually recovers the
true carry-over (adstock) and diminishing-returns (saturation) of each channel.

Run:  python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_WEEKS = 156  # three years of weekly data
OUT = Path(__file__).with_name("marketing_data.csv")

# Ground truth used to simulate sales (the model never sees these values)
TRUE = {
    #            decay  half-sat (k)  shape (s)  max weekly effect ($ sales)
    "tv":       (0.60,   55_000,      2.0,       155_000),
    "search":   (0.10,   28_000,      1.2,       120_000),
    "social":   (0.35,   30_000,      1.5,        70_000),
    "display":  (0.30,   15_000,      1.4,        22_000),
    "email":    (0.20,    3_000,      1.1,        38_000),
}


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for t, v in enumerate(x):
        carry = v + decay * carry
        out[t] = carry
    return out


def hill(x: np.ndarray, k: float, s: float) -> np.ndarray:
    return x**s / (x**s + k**s)


def true_media_effects(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Weekly incremental sales each channel *actually* generated (used only for validation)."""
    return {
        c: TRUE[c][3] * hill(geometric_adstock(df[f"{c}_spend"].to_numpy(float), TRUE[c][0]), TRUE[c][1], TRUE[c][2])
        for c in TRUE
    }


def true_steady_state_sales(channel: str, weekly_spend: float) -> float:
    """True weekly incremental sales from a constant weekly spend (validation only)."""
    decay, k, s, beta = TRUE[channel]
    return float(beta * hill(np.array([weekly_spend / (1 - decay)]), k, s)[0])


def main() -> None:
    rng = np.random.default_rng(SEED)
    weeks = pd.date_range("2023-01-02", periods=N_WEEKS, freq="W-MON")
    t = np.arange(N_WEEKS)
    week_of_year = weeks.isocalendar().week.to_numpy().astype(int)
    annual = np.sin(2 * np.pi * (t - 10) / 52)  # peaks late summer

    # --- media spend (weekly, USD) --------------------------------------
    tv = np.where(rng.random(N_WEEKS) < 0.45, rng.uniform(40_000, 95_000, N_WEEKS), 0.0)
    tv[(week_of_year >= 46) & (week_of_year <= 51)] *= 1.4  # holiday flighting
    # bid tests / budget changes give the model the spend variation it needs to learn from
    search = 26_000 + rng.normal(0, 7_000, N_WEEKS)
    search *= np.repeat(rng.choice([0.6, 1.0, 1.4], size=N_WEEKS // 4 + 1), 4)[:N_WEEKS]
    social = 22_000 + 6_000 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 5_000, N_WEEKS)
    display = 11_000 * np.repeat(rng.choice([0.5, 1.0, 1.5], size=N_WEEKS // 6 + 1), 6)[:N_WEEKS]
    display = display + rng.normal(0, 1_500, N_WEEKS)
    email = rng.uniform(1_500, 5_000, N_WEEKS)
    spend = {
        "tv": tv,
        "search": np.clip(search, 5_000, None),
        "social": np.clip(social, 5_000, None),
        "display": np.clip(display, 2_000, None),
        "email": email,
    }

    # --- controls -------------------------------------------------------
    price_index = 1.0 + 0.04 * np.sin(2 * np.pi * t / 52 + 1) + rng.normal(0, 0.01, N_WEEKS)
    promo = (rng.random(N_WEEKS) < 0.12).astype(int)
    holiday = ((week_of_year >= 47) & (week_of_year <= 52)).astype(int)

    # --- sales ----------------------------------------------------------
    base = 420_000 + 900 * t  # organic growth
    seasonality = 45_000 * annual
    media = sum(
        TRUE[c][3] * hill(geometric_adstock(spend[c], TRUE[c][0]), TRUE[c][1], TRUE[c][2])
        for c in spend
    )
    sales = (
        base
        + seasonality
        + media
        - 380_000 * (price_index - 1.0)
        + 60_000 * promo
        + 95_000 * holiday
        + rng.normal(0, 14_000, N_WEEKS)
    )

    df = pd.DataFrame(
        {
            "week": weeks.date,
            **{f"{c}_spend": np.round(v, 2) for c, v in spend.items()},
            "price_index": np.round(price_index, 4),
            "promo": promo,
            "holiday": holiday,
            "sales": np.round(sales, 2),
        }
    )
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} weeks -> {OUT}")


if __name__ == "__main__":
    main()

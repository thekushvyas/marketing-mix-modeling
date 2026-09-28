"""
End-to-end Marketing Mix Modeling pipeline.

    python data/generate_data.py   # (optional) regenerate the synthetic dataset
    python run_analysis.py         # fit, evaluate, decompose, optimise, and draw charts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from mmm import CHANNELS, MMM, add_controls  # noqa: E402
from optimize_budget import optimise_budget  # noqa: E402
from generate_data import TRUE, true_media_effects, true_steady_state_sales  # noqa: E402

IMG, OUT = ROOT / "images", ROOT / "outputs"
IMG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
HOLDOUT = 12  # last 12 weeks kept out of training

NAMES = {"tv": "TV", "search": "Paid Search", "social": "Social", "display": "Display", "email": "Email"}
COLORS = {"baseline": "#d9dde5", "tv": "#1f3a93", "search": "#2f6fdf", "social": "#6ea8ff",
          "display": "#9aa9c4", "email": "#e0457b"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8f98", "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555",
    "axes.titleweight": "bold", "axes.titlesize": 12, "svg.fonttype": "none",
})


def money(x, _=None):
    return f"${x/1e3:,.0f}K" if abs(x) < 1e6 else f"${x/1e6:,.1f}M"


def mape(a, f):
    return float(np.mean(np.abs((a - f) / a)) * 100)


def r2(a, f):
    return float(1 - np.sum((a - f) ** 2) / np.sum((a - np.mean(a)) ** 2))


def main() -> None:
    df = add_controls(pd.read_csv(ROOT / "data" / "marketing_data.csv", parse_dates=["week"]))
    y = df["sales"].to_numpy(float)
    train, test = df.iloc[:-HOLDOUT], df.iloc[-HOLDOUT:]

    # ---------------- evaluate on a holdout ----------------
    m_eval = MMM().fit(train, y[:-HOLDOUT])
    fit_train, fit_test = m_eval.predict(train), m_eval.predict(test)
    metrics = {
        "train_r2": r2(y[:-HOLDOUT], fit_train), "train_mape_pct": mape(y[:-HOLDOUT], fit_train),
        "holdout_r2": r2(y[-HOLDOUT:], fit_test), "holdout_mape_pct": mape(y[-HOLDOUT:], fit_test),
    }

    # ---------------- refit on all data for decomposition ----------------
    model = MMM().fit(df, y)
    contrib = model.contributions(df)
    fitted = contrib.sum(axis=1).to_numpy()

    spend = {c: float(df[f"{c}_spend"].sum()) for c in CHANNELS}
    effect = {c: float(contrib[c].sum()) for c in CHANNELS}
    avg_week = {c: float(df[f"{c}_spend"].mean()) for c in CHANNELS}
    truth = true_media_effects(df)
    summary = pd.DataFrame({
        "channel": [NAMES[c] for c in CHANNELS],
        "spend": [spend[c] for c in CHANNELS],
        "incremental_sales": [effect[c] for c in CHANNELS],
        "share_of_sales_pct": [100 * effect[c] / y.sum() for c in CHANNELS],
        "roi": [effect[c] / spend[c] for c in CHANNELS],
        "marginal_roi": [model.marginal_roi(c, avg_week[c]) for c in CHANNELS],
        "true_roi": [truth[c].sum() / spend[c] for c in CHANNELS],
        "est_decay": [model.params_[c]["decay"] for c in CHANNELS],
        "true_decay": [TRUE[c][0] for c in CHANNELS],
    })
    summary.round(3).to_csv(OUT / "channel_summary.csv", index=False)

    # ---------------- budget optimisation ----------------
    optimal = optimise_budget(model, avg_week)
    cur_sales = sum(model.response(c, avg_week[c]) for c in CHANNELS)
    opt_sales = sum(model.response(c, optimal[c]) for c in CHANNELS)
    uplift = opt_sales - cur_sales
    # sanity check against the simulator's ground truth: does the new plan *really* sell more?
    true_cur = sum(true_steady_state_sales(c, avg_week[c]) for c in CHANNELS)
    true_opt = sum(true_steady_state_sales(c, optimal[c]) for c in CHANNELS)
    plan = pd.DataFrame({
        "channel": [NAMES[c] for c in CHANNELS],
        "current_weekly_spend": [avg_week[c] for c in CHANNELS],
        "optimal_weekly_spend": [optimal[c] for c in CHANNELS],
        "change_pct": [100 * (optimal[c] / avg_week[c] - 1) for c in CHANNELS],
    })
    plan.round(1).to_csv(OUT / "budget_plan.csv", index=False)

    results = {
        **{k: round(v, 3) for k, v in metrics.items()},
        "media_share_of_sales_pct": round(100 * sum(effect.values()) / y.sum(), 1),
        "weekly_media_budget": round(sum(avg_week.values())),
        "weekly_incremental_sales_current": round(cur_sales),
        "weekly_incremental_sales_optimal": round(opt_sales),
        "weekly_uplift": round(uplift),
        "uplift_pct_of_media_sales": round(100 * uplift / cur_sales, 1),
        "annual_uplift": round(uplift * 52),
        "true_weekly_uplift": round(true_opt - true_cur),
        "true_annual_uplift": round((true_opt - true_cur) * 52),
        "true_media_share_of_sales_pct": round(100 * sum(v.sum() for v in truth.values()) / y.sum(), 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # ================= charts =================
    weeks = df["week"]

    # 1. actual vs fitted with holdout
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(weeks, y, color="#1d1d1f", lw=1.4, label="Actual sales")
    ax.plot(weeks[:-HOLDOUT], fit_train, color="#2f6fdf", lw=1.4, label="Model (train)")
    ax.plot(weeks[-HOLDOUT:], fit_test, color="#e0457b", lw=1.8, label="Model (holdout forecast)")
    ax.axvspan(weeks.iloc[-HOLDOUT], weeks.iloc[-1], color="#e0457b", alpha=0.07)
    ax.yaxis.set_major_formatter(money)
    ax.set_title(f"Actual vs. predicted weekly sales  —  holdout MAPE {metrics['holdout_mape_pct']:.1f}%", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "01_fit.svg"); plt.close(fig)

    # 2. decomposition (4-week averages keep the picture readable)
    blk = np.arange(len(df)) // 4
    agg = contrib.groupby(blk).mean()
    y4 = pd.Series(y).groupby(blk).mean()
    w4 = weeks.groupby(blk).first()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    order = ["baseline"] + CHANNELS
    ax.stackplot(w4, *[agg[c] for c in order], colors=[COLORS[c] for c in order],
                 labels=["Baseline"] + [NAMES[c] for c in CHANNELS], alpha=0.95, lw=0)
    ax.plot(w4, y4, color="#1d1d1f", lw=0.9, label="Actual")
    ax.yaxis.set_major_formatter(money)
    ax.set_title("What drives sales: baseline vs. each channel (4-week averages)", loc="left")
    ax.legend(frameon=False, ncol=7, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, y.max() * 1.18)
    fig.tight_layout(); fig.savefig(IMG / "02_decomposition.svg"); plt.close(fig)

    # 3. ROI
    s = summary.sort_values("roi")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    bars = ax.barh(s["channel"], s["roi"], color=[COLORS[k] for k in
                   [c for n in s["channel"] for c, v in NAMES.items() if v == n]])
    ax.axvline(1, color="#8a8f98", ls="--", lw=1)
    ax.text(1.02, -0.6, "break-even", color="#666", fontsize=8)
    for b, v, mr in zip(bars, s["roi"], s["marginal_roi"]):
        ax.text(b.get_width() + 0.08, b.get_y() + b.get_height() / 2,
                f"\\${v:.2f} per \\$1   (next \\$1 → \\${mr:.2f})", va="center", fontsize=9, color="#333")
    ax.set_xlim(0, s["roi"].max() * 1.6)
    ax.set_title("Return on ad spend by channel (average vs. marginal)", loc="left")
    ax.set_xlabel("Incremental sales per $1 of spend")
    fig.tight_layout(); fig.savefig(IMG / "03_roi.svg"); plt.close(fig)

    # 4. response curves
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for c in CHANNELS:
        grid = np.linspace(0, avg_week[c] * 3, 120)
        ax.plot(grid, model.response(c, grid), color=COLORS[c], lw=2, label=NAMES[c])
        ax.scatter([avg_week[c]], [model.response(c, avg_week[c])], color=COLORS[c], s=28, zorder=3)
    ax.xaxis.set_major_formatter(money); ax.yaxis.set_major_formatter(money)
    ax.set_xlabel("Weekly spend"); ax.set_ylabel("Weekly incremental sales")
    ax.set_title("Diminishing returns: response curves (dot = current spend)", loc="left")
    ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(IMG / "04_response_curves.svg"); plt.close(fig)

    # 5. budget reallocation
    fig, ax = plt.subplots(figsize=(9, 3.2))
    xpos = np.arange(len(CHANNELS)); wbar = 0.38
    ax.bar(xpos - wbar / 2, plan["current_weekly_spend"], wbar, color="#c9ced8", label="Current")
    ax.bar(xpos + wbar / 2, plan["optimal_weekly_spend"], wbar, color="#2f6fdf", label="Optimised")
    for i, pct in enumerate(plan["change_pct"]):
        ax.text(xpos[i] + wbar / 2, plan["optimal_weekly_spend"].iloc[i] * 1.02, f"{pct:+.0f}%",
                ha="center", fontsize=9, color="#333")
    ax.set_xticks(xpos, plan["channel"]); ax.yaxis.set_major_formatter(money)
    ax.set_title(f"Same budget, better mix: +{money(uplift)} incremental sales per week "
                 f"(+{results['uplift_pct_of_media_sales']}%)", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "05_budget_optimisation.svg"); plt.close(fig)

    print(json.dumps(results, indent=2))
    print(summary.round(2).to_string(index=False))
    print(plan.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

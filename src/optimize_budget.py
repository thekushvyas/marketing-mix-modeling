"""Re-allocate a fixed weekly budget across channels to maximise predicted sales."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def optimise_budget(model, current: dict[str, float], lower=0.5, upper=2.0) -> dict[str, float]:
    """Keep total spend fixed; each channel may move between `lower`x and `upper`x its current level."""
    chans = list(current)
    x0 = np.array([current[c] for c in chans])
    total = x0.sum()

    def neg_sales(x):
        return -sum(model.response(c, x[i]) for i, c in enumerate(chans))

    res = minimize(
        neg_sales, x0, method="SLSQP",
        bounds=[(lower * v, upper * v) for v in x0],
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - total}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    return {c: float(res.x[i]) for i, c in enumerate(chans)}

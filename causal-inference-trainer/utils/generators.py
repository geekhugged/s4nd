import numpy as np
import pandas as pd


def ice_cream_drowning(n: int = 300, confounder_strength: float = 3.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    season = rng.uniform(0, 1, n)
    ice_cream = 10 + confounder_strength * 15 * season + rng.normal(0, 3, n)
    drowning = 2 + confounder_strength * 5 * season + rng.normal(0, 1.5, n)
    return pd.DataFrame({
        "season": season,
        "season_label": np.where(season > 0.5, "Лето", "Зима"),
        "ice_cream_sales": np.clip(ice_cream, 0, None),
        "drowning_incidents": np.clip(drowning, 0, None),
    })


def couriers_vs_orders(n: int = 400, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hour = rng.uniform(0, 24, n)
    peak = (np.exp(-0.5 * ((hour - 12) / 4) ** 2) +
            0.8 * np.exp(-0.5 * ((hour - 19) / 2) ** 2))
    couriers = np.clip(50 + 80 * peak + rng.normal(0, 8, n), 5, None)
    orders = np.clip(100 + 200 * peak + rng.normal(0, 20, n), 0, None)
    period = np.where(hour < 11, "Утро", np.where(hour < 17, "День", "Вечер"))
    return pd.DataFrame({
        "hour": hour.round(1),
        "period": period,
        "couriers": couriers.round().astype(int),
        "orders": orders.round().astype(int),
    })


def promo_with_selection_bias(n: int = 1000, true_effect: float = 2.0, seed: int = 42) -> pd.DataFrame:
    """
    True causal effect of promo = true_effect extra orders.
    But promo is given to more active users → naive estimate is biased upward.
    """
    rng = np.random.default_rng(seed)
    activity = rng.normal(0, 1, n)
    promo_prob = 1 / (1 + np.exp(-activity))
    promo = rng.binomial(1, promo_prob, n)
    y0 = np.clip(5 + 3 * activity + rng.normal(0, 1, n), 0, None)
    y1 = y0 + true_effect
    y_obs = promo * y1 + (1 - promo) * y0
    return pd.DataFrame({
        "activity": activity.round(3),
        "promo": promo,
        "orders": np.maximum(0, y_obs).round().astype(int),
        "y0_true": np.maximum(0, y0).round().astype(int),
        "y1_true": np.maximum(0, y1).round().astype(int),
    })


def did_experiment(n_cities: int = 20, n_periods: int = 8,
                   treatment_period: int = 4, true_effect: float = 15.0,
                   seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for city in range(n_cities):
        treated = city < n_cities // 2
        city_fe = rng.normal(0, 10)
        for t in range(n_periods):
            post = t >= treatment_period
            effect = true_effect if (treated and post) else 0
            orders = 100 + city_fe + 5 * t + effect + rng.normal(0, 8)
            rows.append({
                "city": city, "period": t,
                "treated": int(treated), "post": int(post),
                "orders": max(0, orders),
            })
    return pd.DataFrame(rows)


def ab_test_data(n_per_group: int = 500, true_effect: float = 1.5,
                 baseline: float = 8.0, std: float = 4.0,
                 seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    control = rng.normal(baseline, std, n_per_group)
    treatment = rng.normal(baseline + true_effect, std, n_per_group)
    df = pd.DataFrame({
        "orders": np.concatenate([control, treatment]),
        "group": ["Контроль"] * n_per_group + ["Тест"] * n_per_group,
    })
    df["orders"] = np.maximum(0, df["orders"].round()).astype(int)
    return df


def matching_data(n: int = 1000, true_effect: float = 3.0, seed: int = 42) -> pd.DataFrame:
    """
    Observational promo study with two confounders:
    orders_history and days_since_install.
    Heavy users are more likely to receive promo.
    """
    rng = np.random.default_rng(seed)
    orders_history = rng.poisson(8, n).astype(float)
    days_since_install = rng.exponential(90, n)

    logit = -1.5 + 0.15 * orders_history - 0.005 * days_since_install
    promo_prob = 1 / (1 + np.exp(-logit))
    promo = rng.binomial(1, promo_prob, n)

    y0 = np.clip(2 + 0.5 * orders_history - 0.01 * days_since_install
                 + rng.normal(0, 1.5, n), 0, None)
    y1 = y0 + true_effect
    y_obs = promo * y1 + (1 - promo) * y0

    return pd.DataFrame({
        "orders_history": orders_history.round().astype(int),
        "days_since_install": days_since_install.round().astype(int),
        "promo": promo,
        "orders_next_week": np.maximum(0, y_obs).round().astype(int),
        "y0_true": np.maximum(0, y0).round().astype(int),
        "y1_true": np.maximum(0, y1).round().astype(int),
    })

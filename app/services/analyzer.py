from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_basic_stats(df: pd.DataFrame) -> Dict:
    """
    Compute basic income/expense stats for the given period.
    Expects columns: ['amount', 'is_income'].
    """
    income = float(df.loc[df["is_income"], "amount"].sum())
    expense = float(df.loc[~df["is_income"], "amount"].sum())
    net = income - expense
    savings_rate = (net / income) if income > 0 else 0.0

    return {
        "total_income": round(income, 2),
        "total_expense": round(expense, 2),
        "net": round(net, 2),
        "savings_rate": round(savings_rate, 3),
    }


def compute_category_breakdown(df: pd.DataFrame) -> List[Dict]:
    """
    Return top spending categories sorted by expense (desc).
    Only considers expenses (is_income == False).
    """
    if "category" not in df.columns:
        return []

    exp_df = df[~df["is_income"]].copy()
    if exp_df.empty:
        return []

    grouped = exp_df.groupby("category")["amount"].sum().sort_values(ascending=False)

    result = [
        {"category": str(cat), "total_spend": round(float(amount), 2)}
        for cat, amount in grouped.items()
    ]
    return result


def detect_patterns(df: pd.DataFrame) -> Dict:
    """
    Detect a few 'bad' patterns in spending:
    - high fees / charges
    - impulse spending spikes
    - subscription bloat
    - overall cashflow health flag
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1) High fees / charges
    text = df["description"].astype(str).str.lower()
    fees_mask = (
        text.str.contains("fee")
        | text.str.contains("charge")
        | text.str.contains("penalty")
        | text.str.contains("fine")
        | text.str.contains("interest")
    ) & (~df["is_income"])

    fees_total = float(df.loc[fees_mask, "amount"].sum())
    fees_count = int(fees_mask.sum())

    # 2) Impulse spikes: days where expenses > 1.5x median daily expense
    expense_df = df[~df["is_income"]].copy()
    expense_df["date"] = expense_df["timestamp"].dt.date

    impulse_days = []
    extra_spend = 0.0
    if not expense_df.empty:
        daily = expense_df.groupby("date")["amount"].sum()
        median_daily = float(daily.median()) if len(daily) > 0 else 0.0

        if median_daily > 0:
            threshold = 1.5 * median_daily
            high_days = daily[daily > threshold]
            impulse_days = [str(d) for d in high_days.index]

            # Extra spend above median on those days
            extra_spend = float((high_days - median_daily).sum())

    # 3) Subscription bloat
    cat_col = df.get("category", pd.Series([None] * len(df)))
    cat_str = cat_col.astype(str).str.lower()
    subs_mask = ((cat_str == "subscriptions") | text.str.contains("subscription")) & (
        ~df["is_income"]
    )

    subs_total = float(df.loc[subs_mask, "amount"].sum())
    subs_count = int(subs_mask.sum())

    # 4) Cashflow health
    stats = compute_basic_stats(df)
    net = stats["net"]
    savings_rate = stats["savings_rate"]

    if net < 0:
        cashflow_flag = "critical"
    elif savings_rate < 0.1:
        cashflow_flag = "warning"
    else:
        cashflow_flag = "ok"

    return {
        "high_fees": {
            "count": fees_count,
            "total": round(fees_total, 2),
        },
        "impulse_spikes": {
            "days": impulse_days,
            "extra_spend": round(extra_spend, 2),
        },
        "subscriptions": {
            "count": subs_count,
            "total": round(subs_total, 2),
        },
        "cashflow_flag": cashflow_flag,
    }


def build_behaviour_profiles(
    df_all: pd.DataFrame, n_clusters: int = 3
) -> Dict[str, Dict]:
    """
    Cluster behaviour across months into simple profiles, e.g.:
    - 'Super Saver'
    - 'Balanced'
    - 'Subscription Heavy'
    - 'Lifestyle Spender'

    Returns:
        {
          "labels_by_period": { "2025-01": "Super Saver", ... },
          "cluster_descriptions": { "Super Saver": "...", ... }
        }
    """
    if df_all.empty:
        return {"labels_by_period": {}, "cluster_descriptions": {}}

    df = df_all.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["period"] = df["timestamp"].dt.to_period("M").astype(str)

    rows = []
    for period, group in df.groupby("period"):
        stats = compute_basic_stats(group)

        total_expense = stats["total_expense"]
        total_expense = total_expense if total_expense > 0 else 1e-6

        # Subscription share
        text = group["description"].astype(str).str.lower()
        cat = group.get("category", pd.Series([None] * len(group))).astype(str).str.lower()

        subs_mask = ((cat == "subscriptions") | text.str.contains("subscription")) & (
            ~group["is_income"]
        )
        subs_spend = float(group.loc[subs_mask, "amount"].sum())
        subs_share = subs_spend / total_expense

        # Shopping share
        shop_mask = (cat == "shopping") & (~group["is_income"])
        shop_spend = float(group.loc[shop_mask, "amount"].sum())
        shop_share = shop_spend / total_expense

        rows.append(
            {
                "period": period,
                "savings_rate": stats["savings_rate"],
                "total_expense": stats["total_expense"],
                "subs_share": subs_share,
                "shopping_share": shop_share,
            }
        )

    monthly_df = pd.DataFrame(rows)
    if monthly_df.empty:
        return {"labels_by_period": {}, "cluster_descriptions": {}}

    # If too few months, just heuristic labels instead of clustering
    if len(monthly_df) < 2:
        labels = {}
        for _, row in monthly_df.iterrows():
            if row["savings_rate"] >= 0.4:
                label = "Super Saver"
            elif row["subs_share"] > 0.2:
                label = "Subscription Heavy"
            elif row["shopping_share"] > 0.25:
                label = "Lifestyle Spender"
            else:
                label = "Balanced"
            labels[row["period"]] = label

        cluster_descriptions = _default_cluster_descriptions()
        return {
            "labels_by_period": labels,
            "cluster_descriptions": cluster_descriptions,
        }

    feature_cols = ["savings_rate", "total_expense", "subs_share", "shopping_share"]
    X = monthly_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = min(n_clusters, len(monthly_df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    monthly_df["cluster"] = kmeans.fit_predict(X_scaled)

    # Compute cluster means to decide labels
    cluster_info = []
    for cluster_id, grp in monthly_df.groupby("cluster"):
        cluster_info.append(
            {
                "cluster": cluster_id,
                "mean_savings": grp["savings_rate"].mean(),
                "mean_expense": grp["total_expense"].mean(),
                "mean_subs_share": grp["subs_share"].mean(),
                "mean_shop_share": grp["shopping_share"].mean(),
            }
        )

    # Sort clusters for each dimension
    # We'll assign labels based on dominant characteristic.
    remaining_ids = {c["cluster"] for c in cluster_info}
    cluster_labels: Dict[int, str] = {}

    # 1) Super Saver: highest savings_rate
    saver_cluster = max(cluster_info, key=lambda c: c["mean_savings"])["cluster"]
    cluster_labels[saver_cluster] = "Super Saver"
    remaining_ids.discard(saver_cluster)

    # 2) Subscription Heavy: highest subs_share among remaining
    if remaining_ids:
        subs_cluster = max(
            [c for c in cluster_info if c["cluster"] in remaining_ids],
            key=lambda c: c["mean_subs_share"],
        )["cluster"]
        cluster_labels[subs_cluster] = "Subscription Heavy"
        remaining_ids.discard(subs_cluster)

    # 3) Lifestyle Spender: highest shopping_share among remaining
    if remaining_ids:
        shop_cluster = max(
            [c for c in cluster_info if c["cluster"] in remaining_ids],
            key=lambda c: c["mean_shop_share"],
        )["cluster"]
        cluster_labels[shop_cluster] = "Lifestyle Spender"
        remaining_ids.discard(shop_cluster)

    # 4) Everything else → Balanced
    for cid in remaining_ids:
        cluster_labels[cid] = "Balanced"

    labels_by_period: Dict[str, str] = {}
    for _, row in monthly_df.iterrows():
        c = int(row["cluster"])
        labels_by_period[row["period"]] = cluster_labels.get(c, "Balanced")

    cluster_descriptions = _default_cluster_descriptions()

    return {
        "labels_by_period": labels_by_period,
        "cluster_descriptions": cluster_descriptions,
    }


def _default_cluster_descriptions() -> Dict[str, str]:
    return {
        "Super Saver": "High savings rate, controlled expenses. You prioritise surplus and future security.",
        "Balanced": "Reasonable spending and decent savings. You’re mostly in control with room to optimise.",
        "Subscription Heavy": "Noticeable chunk of spend goes into recurring services. Good candidate for pruning.",
        "Lifestyle Spender": "Higher shopping / lifestyle spend. You enjoy comforts, but may trim a bit without pain.",
    }

def compute_category_trends(
    df_all: pd.DataFrame,
    current_period: str,
) -> List[Dict]:
    """
    Compare current period's category spend vs average monthly spend in other periods.

    Returns a list of:
      {
        "category": str,
        "current": float,
        "baseline": float,
        "delta": float,
        "delta_pct": float | None
      }

    Sorted by absolute percentage change (largest movers first).
    """
    df = df_all.copy()
    if "period" not in df.columns:
        # Expect 'period' column to exist (YYYY-MM)
        return []

    # Only expenses
    df = df[~df["is_income"]].copy()

    cur = df[df["period"] == current_period]
    base = df[df["period"] != current_period]

    if cur.empty or base.empty:
        return []

    # Current period spend by category
    cur_by_cat = cur.groupby("category")["amount"].sum()

    # Baseline: avg monthly spend per category over other periods
    base["period_cat"] = base["period"] + "|" + base["category"].astype(str)
    base_by_period_cat = base.groupby(["period", "category"])["amount"].sum().reset_index()
    base_months = base["period"].nunique()

    if base_months == 0:
        return []

    baseline_by_cat = (
        base_by_period_cat.groupby("category")["amount"].sum() / base_months
    )

    cats = set(cur_by_cat.index) | set(baseline_by_cat.index)

    rows: List[Dict] = []
    for cat in cats:
        current = float(cur_by_cat.get(cat, 0.0))
        baseline = float(baseline_by_cat.get(cat, 0.0))
        delta = current - baseline
        if baseline > 0:
            delta_pct = delta / baseline
        else:
            delta_pct = None

        rows.append(
            {
                "category": str(cat),
                "current": round(current, 2),
                "baseline": round(baseline, 2),
                "delta": round(delta, 2),
                "delta_pct": round(delta_pct, 3) if delta_pct is not None else None,
            }
        )

    # Sort by biggest percentage movers
    def sort_key(row):
        if row["delta_pct"] is None:
            return 0.0
        return abs(row["delta_pct"])

    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    return rows_sorted
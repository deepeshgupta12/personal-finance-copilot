from typing import Dict, List


def recommend_actions_for_next_month(
    stats: Dict, categories: List[Dict], patterns: Dict
) -> List[str]:
    actions: List[str] = []

    savings_rate = stats.get("savings_rate", 0.0)
    net = stats.get("net", 0.0)
    income = stats.get("total_income", 0.0)
    expense = stats.get("total_expense", 0.0)

    fees = patterns.get("high_fees", {})
    subs = patterns.get("subscriptions", {})
    spikes = patterns.get("impulse_spikes", {})

    fees_total = fees.get("total", 0.0)
    fees_count = fees.get("count", 0)
    subs_total = subs.get("total", 0.0)
    subs_count = subs.get("count", 0)
    spike_days = spikes.get("days", [])
    extra_spend = spikes.get("extra_spend", 0.0)

    # 1) Savings baseline
    if savings_rate < 0.1:
        actions.append(
            "Set up an automatic transfer of at least 10% of your income into a separate savings or investment account at the start of the month."
        )
    elif savings_rate < 0.25:
        actions.append(
            "If it feels comfortable, increase your monthly savings by 5–10% of your income by slightly trimming non-essential spends."
        )
    else:
        actions.append(
            "Your savings rate is strong. Protect it by keeping fixed essentials and recurring commitments stable."
        )

    # 2) Fees / charges
    if fees_total > 0:
        actions.append(
            f"Open your statement and identify the {fees_count} fee/charge transaction(s) (~₹{fees_total:,.0f}). Decide which ones you can avoid next month (late fees, ATM fees, convenience charges)."
        )

    # 3) Impulse spikes
    if spike_days and extra_spend > 0:
        actions.append(
            f"Pick one of the high-spend days from this month (e.g., {spike_days[0]}). Plan in advance how you'll handle a similar day next month to avoid the extra ~₹{extra_spend:,.0f} spend."
        )

    # 4) Subscriptions
    if subs_total > 0:
        actions.append(
            f"Do a 10-minute subscription audit. Keep what you use weekly and pause or cancel the rest to reclaim part of the ~₹{subs_total:,.0f} you spent this month."
        )

    # 5) Top category nudge
    if categories:
        top_cat = categories[0]
        if top_cat.get("category"):
            actions.append(
                f"Choose one simple rule to soften {top_cat['category']} spends next month—for example, one fewer order or a fixed monthly cap."
            )

    # Guarantee at least 3 actions
    if len(actions) < 3:
        actions.append(
            "Block 30 minutes on your calendar to review this month’s spending and write down 1–2 habits you want to change. Treat it like a personal retro, not punishment."
        )

    # Deduplicate while preserving order, limit to 5
    seen = set()
    deduped: List[str] = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            deduped.append(a)

    return deduped[:5]
import os
from textwrap import dedent
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Try loading .env (OPENAI_API_KEY, etc.)
load_dotenv()


def _template_story(
    period: str,
    stats: Dict,
    categories: List[Dict],
    patterns: Dict,
) -> str:
    """
    Deterministic, non-LLM fallback story in simple language.
    """
    income = stats["total_income"]
    expense = stats["total_expense"]
    net = stats["net"]
    savings_rate = stats["savings_rate"]

    # Top 3 categories by spend
    top_cats = categories[:3]
    if top_cats:
        cats_text = "; ".join(
            f"{c['category']} ({c['total_spend']:.0f})" for c in top_cats
        )
    else:
        cats_text = "No major expenses recorded."

    fees = patterns["high_fees"]
    spikes = patterns["impulse_spikes"]
    subs = patterns["subscriptions"]
    flag = patterns["cashflow_flag"]

    tone = "steady and healthy overall."
    if flag == "critical":
        tone = "at risk — your spending was higher than your income."
    elif flag == "warning":
        tone = "okay, but your savings cushion is thinner than ideal."

    story = f"""
    For {period}, here’s your money story in plain language:

    You brought in about ₹{income:,.0f} and spent around ₹{expense:,.0f}.
    That leaves you with roughly ₹{net:,.0f} left over, which means your savings rate
    for the month was about {savings_rate * 100:.1f}%. Overall, your cashflow looks {tone}

    Most of your outgoing money went into: {cats_text}

    Looking at patterns, you paid about ₹{fees['total']:,.0f} in fees or charges
    across {fees['count']} transaction(s). This is money that gives you zero value
    back — it’s usually worth reducing this over time.

    We also noticed {len(spikes['days'])} day(s) where your spending spiked well above
    your usual daily level. On those days you spent roughly ₹{spikes['extra_spend']:,.0f}
    more than your typical pattern. These are good candidates for “impulse days” to
    reflect on.

    Subscriptions (OTT, music, etc.) added up to about ₹{subs['total']:,.0f}
    across {subs['count']} transaction(s). It might be worth checking if all of them
    are still actively used.

    If you want one simple takeaway: reduce unnecessary fees and tame 1–2 of the
    spike days next month — that alone can improve your savings rate meaningfully
    without feeling like a harsh budget.
    """
    return dedent(story).strip()


def _llm_story(
    period: str,
    stats: Dict,
    categories: List[Dict],
    patterns: Dict,
) -> Optional[str]:
    """
    Optional LLM-based story using OpenAI Chat Completions.
    Returns None if anything fails (no key, model not found, etc.).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a friendly, non-judgmental personal finance coach for young "
        "professionals in India."
    )

    user_prompt = f"""
    You are given:
    - A month label: {period}
    - Basic stats: {stats}
    - Top spending categories: {categories}
    - Detected patterns: {patterns}

    Task:
    - Write a conversational, empathetic "money story" for this month.
    - Use very simple language and short paragraphs.
    - Do NOT mention that you used stats or data; just talk like a coach.
    - Include:
      - How much they roughly earned and spent
      - Biggest spending areas in plain language
      - A gentle comment on savings health (not shaming)
      - 2–3 concrete, realistic suggestions they can try next month
    - Keep it under 300 words.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-5",  # <-- CHANGE THIS if you prefer another model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = completion.choices[0].message.content
        return text.strip() if text else None
    except Exception as e:
        # Optional: log to console for debugging
        print(f"[LLM STORY ERROR] {e}")
        return None


def build_money_story(
    period: str,
    stats: Dict,
    categories: List[Dict],
    patterns: Dict,
) -> str:
    """
    Try LLM-based story first; if unavailable, use the local template.
    """
    llm_text = _llm_story(period, stats, categories, patterns)
    if llm_text:
        return llm_text

    return _template_story(period, stats, categories, patterns)
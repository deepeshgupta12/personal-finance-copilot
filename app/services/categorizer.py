from typing import Optional

import pandas as pd


# Simple keyword-based categorisation.
# This is deliberately opinionated for young professionals in India.
KEYWORD_CATEGORY_MAP = {
    # Food & eating out
    "zomato": "Food",
    "swiggy": "Food",
    "blinkit": "Food",
    "instamart": "Food",
    "bigbasket": "Food",
    "starbucks": "Food",
    "cafe": "Food",
    "restaurant": "Food",

    # Transport
    "uber": "Transport",
    "ola": "Transport",
    "rapido": "Transport",
    "metro": "Transport",
    "cab": "Transport",
    "fuel": "Transport",
    "petrol": "Transport",
    "diesel": "Transport",

    # Subscriptions / OTT
    "netflix": "Subscriptions",
    "spotify": "Subscriptions",
    "hotstar": "Subscriptions",
    "disney": "Subscriptions",
    "prime": "Subscriptions",
    "youtube": "Subscriptions",
    "apple music": "Subscriptions",

    # Shopping / lifestyle
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "myntra": "Shopping",
    "ajio": "Shopping",
    "h&m": "Shopping",
    "zara": "Shopping",
    "nykaa": "Shopping",

    # Housing
    "rent": "Housing",
    "landlord": "Housing",
    "maintenance": "Housing",
    "society": "Housing",

    # Income
    "salary": "Salary",
    "salaried": "Salary",
    "stipend": "Salary",
    "freelance": "Side Income",
    "consulting": "Side Income",
    "bonus": "Bonus",

    # Fees / charges
    "fee": "Fees & Charges",
    "charges": "Fees & Charges",
    "penalty": "Fees & Charges",
    "fine": "Fees & Charges",
    "interest": "Fees & Charges",
}


def guess_category(
    description: str,
    source: Optional[str] = None,
    account_name: Optional[str] = None,
) -> Optional[str]:
    """
    Very simple heuristic categoriser based on keywords.
    Returns a category string or None if no match.
    """
    text = " ".join(
        part.lower()
        for part in [description, source, account_name]
        if part is not None
    )

    for keyword, category in KEYWORD_CATEGORY_MAP.items():
        if keyword in text:
            return category

    # Fallbacks: if we can’t confidently guess, leave as None.
    return None


def apply_auto_categories_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing / empty categories in a transactions DataFrame.
    Expects columns: ['description', 'source', 'account_name', 'category'].
    """
    if "category" not in df.columns:
        df["category"] = None

    df = df.copy()

    mask_missing = df["category"].isna() | (df["category"].astype(str).str.strip() == "")

    def _categorise_row(row):
        cat = guess_category(
            description=str(row.get("description", "")),
            source=row.get("source"),
            account_name=row.get("account_name"),
        )
        return cat if cat is not None else row.get("category")

    df.loc[mask_missing, "category"] = df.loc[mask_missing].apply(
        _categorise_row, axis=1
    )

    return df
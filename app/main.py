from typing import List

import pandas as pd
import io
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime

from . import models, schemas
from .database import SessionLocal, engine

from .services.categorizer import apply_auto_categories_to_df
from .services.analyzer import (
    compute_basic_stats,
    compute_category_breakdown,
    detect_patterns,
    build_behaviour_profiles,
    compute_category_trends,
)
from .services.storyteller import build_money_story
from .services.actions import recommend_actions_for_next_month

# Create tables (for now we do this on import; later we can move to Alembic)
models.Base.metadata.create_all(bind=engine)

def seed_initial_data():
    db = SessionLocal()
    try:
        # Default demo user
        if db.query(models.User).count() == 0:
            demo = models.User(name="Demo User", email="demo@example.com")
            db.add(demo)
            db.commit()
            db.refresh(demo)
        else:
            demo = db.query(models.User).first()

        # Default budgets for the demo user
        from sqlalchemy import func

        existing_budgets = (
            db.query(models.Budget)
            .filter(models.Budget.user_id == demo.id)
            .count()
        )
        if existing_budgets == 0:
            defaults = [
                ("Food", 5000.0),
                ("Shopping", 4000.0),
                ("Subscriptions", 1500.0),
                ("Transport", 3000.0),
            ]
            for cat, amt in defaults:
                db.add(
                    models.Budget(
                        user_id=demo.id,
                        category=cat,
                        amount=amt,
                    )
                )
            db.commit()
    finally:
        db.close()


seed_initial_data()

app = FastAPI(title="Money Copilot")

# Serve static files (we'll use this later for CSS/JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


def get_db():
    """Dependency that provides a SQLAlchemy session to routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def index(request: Request):
    """
    Simple placeholder homepage.
    We'll later show monthly 'money story', spending patterns, etc.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Money Copilot"},
    )


# ---- Transaction APIs ----

@app.post(
    "/transactions",
    response_model=schemas.Transaction,
    tags=["transactions"],
)
def create_transaction(
    tx: schemas.TransactionCreate,
    db: Session = Depends(get_db),
):
    """Create a single transaction (we'll add CSV upload later)."""
    db_tx = models.Transaction(
        timestamp=tx.timestamp,
        amount=tx.amount,
        currency=tx.currency,
        description=tx.description,
        source=tx.source,
        account_name=tx.account_name,
        is_income=tx.is_income,
        category=tx.category,
    )
    db.add(db_tx)
    db.commit()
    db.refresh(db_tx)
    return db_tx


@app.get(
    "/transactions",
    response_model=List[schemas.Transaction],
    tags=["transactions"],
)
def list_transactions(
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List recent transactions (latest first)."""
    transactions = (
        db.query(models.Transaction)
        .order_by(models.Transaction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return transactions

@app.post("/transactions/import-csv", tags=["transactions"])
async def import_transactions_csv(
    file: UploadFile = File(...),
    user_id: int = Query(1),
    db: Session = Depends(get_db),
):
    """
    Import transactions from a CSV and attach them to a given user_id.

    Expected columns in CSV:
      - timestamp
      - amount
      - is_income
      - category (optional)
      - description (optional)
      - source (optional)
      - account_name (optional)
    Any extra columns (e.g., currency) are ignored.
    """
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read CSV file.")

    required = ["timestamp", "amount", "is_income"]
    for col in required:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required column in CSV: {col}",
            )

    # Normalize types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = pd.to_numeric(df["amount"])
    # If is_income is 0/1 or True/False in CSV, this will normalise it
    df["is_income"] = df["is_income"].astype(bool)

    # Helper to safely get optional fields
    def safe_get(row, col):
        if col in row and pd.notna(row[col]):
            return row[col]
        return None

    imported = 0
    for _, row in df.iterrows():
        tx = models.Transaction(
            timestamp=row["timestamp"].to_pydatetime()
            if isinstance(row["timestamp"], pd.Timestamp)
            else row["timestamp"],
            amount=float(row["amount"]),
            is_income=bool(row["is_income"]),
            category=safe_get(row, "category"),
            description=safe_get(row, "description"),
            source=safe_get(row, "source"),
            account_name=safe_get(row, "account_name"),
            user_id=user_id,
        )
        db.add(tx)
        imported += 1

    db.commit()
    return {"status": "ok", "imported": imported, "user_id": user_id}


# ---- Summary APIs ----

@app.get(
    "/summary/monthly",
    response_model=List[schemas.MonthlySummary],
    tags=["summary"],
)
def monthly_summary(db: Session = Depends(get_db)):
    """
    Return monthly income/expense/net summary.
    period format: 'YYYY-MM'
    """
    transactions = db.query(models.Transaction).all()
    if not transactions:
        return []

    data = [
        {
            "timestamp": t.timestamp,
            "amount": t.amount,
            "is_income": t.is_income,
        }
        for t in transactions
    ]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["period"] = df["timestamp"].dt.to_period("M").astype(str)

    income = df[df["is_income"]].groupby("period")["amount"].sum()
    expense = df[~df["is_income"]].groupby("period")["amount"].sum()

    periods = sorted(set(income.index).union(set(expense.index)))

    result = []
    for p in periods:
        total_income = float(income.get(p, 0.0))
        total_expense = float(expense.get(p, 0.0))
        net = total_income - total_expense
        result.append(
            schemas.MonthlySummary(
                period=p,
                total_income=total_income,
                total_expense=total_expense,
                net=net,
            )
        )

    return result


@app.get(
    "/summary/weekly",
    response_model=List[schemas.WeeklySummary],
    tags=["summary"],
)
def weekly_summary(db: Session = Depends(get_db)):
    """
    Return weekly income/expense/net summary.
    period format: 'YYYY-Www' (ISO week)
    """
    transactions = db.query(models.Transaction).all()
    if not transactions:
        return []

    data = [
        {
            "timestamp": t.timestamp,
            "amount": t.amount,
            "is_income": t.is_income,
        }
        for t in transactions
    ]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    iso = df["timestamp"].dt.isocalendar()
    df["period"] = (
        iso["year"].astype(str)
        + "-W"
        + iso["week"].astype(str).str.zfill(2)
    )

    income = df[df["is_income"]].groupby("period")["amount"].sum()
    expense = df[~df["is_income"]].groupby("period")["amount"].sum()

    periods = sorted(set(income.index).union(set(expense.index)))

    result = []
    for p in periods:
        total_income = float(income.get(p, 0.0))
        total_expense = float(expense.get(p, 0.0))
        net = total_income - total_expense
        result.append(
            schemas.WeeklySummary(
                period=p,
                total_income=total_income,
                total_expense=total_expense,
                net=net,
            )
        )

    return result


@app.get(
    "/analysis/monthly-story",
    tags=["analysis"],
)
def monthly_money_story(
    year: int,
    month: int,
    db: Session = Depends(get_db),
):
    """
    Generate a 'money story' for a given month (YYYY, month).

    Example:
    GET /analysis/monthly-story?year=2025&month=1
    """
    # Compute date range [start, end)
    try:
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid year/month.")

    # Load transactions for this month
    txs = (
        db.query(models.Transaction)
        .filter(models.Transaction.timestamp >= start)
        .filter(models.Transaction.timestamp < end)
        .all()
    )

    if not txs:
        raise HTTPException(
            status_code=404,
            detail="No transactions found for this period.",
        )

    data = [
        {
            "timestamp": t.timestamp,
            "amount": t.amount,
            "is_income": t.is_income,
            "category": t.category,
            "description": t.description,
            "source": t.source,
            "account_name": t.account_name,
        }
        for t in txs
    ]
    df = pd.DataFrame(data)

    # Auto-fill missing categories
    df = apply_auto_categories_to_df(df)

    # Compute stats + patterns
    stats = compute_basic_stats(df)
    categories = compute_category_breakdown(df)
    patterns = detect_patterns(df)

    period_label = f"{year}-{month:02d}"
    story = build_money_story(period_label, stats, categories, patterns)

    return {
        "period": period_label,
        "stats": stats,
        "top_categories": categories,
        "patterns": patterns,
        "story": story,
    }

@app.get(
    "/analysis/actions-next-month",
    tags=["analysis"],
)
def actions_next_month(
    year: int,
    month: int,
    db: Session = Depends(get_db),
):
    """
    Return a suggested action list for what to change next month,
    based on this month's stats and patterns.
    """
    # Compute date range [start, end)
    try:
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid year/month.")

    txs = (
        db.query(models.Transaction)
        .filter(models.Transaction.timestamp >= start)
        .filter(models.Transaction.timestamp < end)
        .all()
    )

    if not txs:
        raise HTTPException(
            status_code=404,
            detail="No transactions found for this period.",
        )

    data = [
        {
            "timestamp": t.timestamp,
            "amount": t.amount,
            "is_income": t.is_income,
            "category": t.category,
            "description": t.description,
            "source": t.source,
            "account_name": t.account_name,
        }
        for t in txs
    ]
    df = pd.DataFrame(data)
    df = apply_auto_categories_to_df(df)

    stats = compute_basic_stats(df)
    categories = compute_category_breakdown(df)
    patterns = detect_patterns(df)
    actions = recommend_actions_for_next_month(stats, categories, patterns)

    period_label = f"{year}-{month:02d}"

    return {
        "period": period_label,
        "stats": stats,
        "patterns": patterns,
        "actions": actions,
    }

@app.get("/dashboard", response_class=HTMLResponse, tags=["ui"])
async def dashboard(
    request: Request,
    user_id: int | None = Query(None),
    year: int | None = Query(None),
    month: int | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Server-side rendered dashboard.
    - Multi-user: user_id selector
    - Month selector: defaults to latest month for that user
    - Behaviour profile across months
    - Category trends
    - Budgets vs actual
    """
    # Load users
    users = db.query(models.User).order_by(models.User.id).all()
    if not users:
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "title": "Money Copilot – Dashboard",
                "no_data": True,
                "no_period_data": False,
                "period": None,
                "available_periods": [],
                "user_list": [],
                "selected_user_id": None,
                "stats": None,
                "top_categories": [],
                "patterns": None,
                "story": "",
                "persona_label": None,
                "persona_description": None,
                "actions": [],
                "category_trends": [],
                "budget_view": [],
            },
        )

    # Determine selected user
    if user_id is not None and any(u.id == user_id for u in users):
        selected_user = next(u for u in users if u.id == user_id)
    else:
        selected_user = users[0]
        user_id = selected_user.id

    # Load all transactions for this user
    all_txs = (
        db.query(models.Transaction)
        .filter(models.Transaction.user_id == selected_user.id)
        .all()
    )
    if not all_txs:
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "title": "Money Copilot – Dashboard",
                "no_data": True,
                "no_period_data": False,
                "period": None,
                "available_periods": [],
                "user_list": users,
                "selected_user_id": selected_user.id,
                "stats": None,
                "top_categories": [],
                "patterns": None,
                "story": "",
                "persona_label": None,
                "persona_description": None,
                "actions": [],
                "category_trends": [],
                "budget_view": [],
            },
        )

    data_all = [
        {
            "timestamp": t.timestamp,
            "amount": t.amount,
            "is_income": t.is_income,
            "category": t.category,
            "description": t.description,
            "source": t.source,
            "account_name": t.account_name,
            "user_id": t.user_id,
        }
        for t in all_txs
    ]
    df_all = pd.DataFrame(data_all)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all = apply_auto_categories_to_df(df_all)
    df_all["period"] = df_all["timestamp"].dt.to_period("M").astype(str)

    available_periods = sorted(df_all["period"].unique())
    if not available_periods:
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "title": "Money Copilot – Dashboard",
                "no_data": True,
                "no_period_data": False,
                "period": None,
                "available_periods": [],
                "user_list": users,
                "selected_user_id": selected_user.id,
                "stats": None,
                "top_categories": [],
                "patterns": None,
                "story": "",
                "persona_label": None,
                "persona_description": None,
                "actions": [],
                "category_trends": [],
                "budget_view": [],
            },
        )

    # Determine selected period
    if year is not None and month is not None:
        period_label = f"{year}-{month:02d}"
    else:
        period_label = available_periods[-1]  # latest

    has_period_data = period_label in available_periods

    # Behaviour profiles across months (for this user)
    profiles = build_behaviour_profiles(df_all)
    labels_by_period = profiles.get("labels_by_period", {})
    cluster_descriptions = profiles.get("cluster_descriptions", {})

    persona_label = labels_by_period.get(period_label)
    persona_description = (
        cluster_descriptions.get(persona_label) if persona_label else None
    )

    if not has_period_data:
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "title": "Money Copilot – Dashboard",
                "no_data": False,
                "no_period_data": True,
                "period": period_label,
                "available_periods": available_periods,
                "user_list": users,
                "selected_user_id": selected_user.id,
                "stats": None,
                "top_categories": [],
                "patterns": None,
                "story": "",
                "persona_label": persona_label,
                "persona_description": persona_description,
                "actions": [],
                "category_trends": [],
                "budget_view": [],
            },
        )

    # Filter to selected period
    df_month = df_all[df_all["period"] == period_label].copy()
    stats = compute_basic_stats(df_month)
    categories = compute_category_breakdown(df_month)
    patterns = detect_patterns(df_month)
    story = build_money_story(period_label, stats, categories, patterns)
    actions = recommend_actions_for_next_month(stats, categories, patterns)

    # Category trends
    category_trends = compute_category_trends(df_all, period_label)

    # Budgets vs actual for this period
    budgets = (
        db.query(models.Budget)
        .filter(models.Budget.user_id == selected_user.id)
        .all()
    )
    expense_df = df_month[~df_month["is_income"]].copy()
    actual_by_cat = (
        expense_df.groupby("category")["amount"].sum().to_dict()
        if not expense_df.empty
        else {}
    )

    budget_view = []
    for b in budgets:
        actual = float(actual_by_cat.get(b.category, 0.0))
        budget_amount = float(b.amount)
        util = actual / budget_amount if budget_amount > 0 else 0.0

        if util < 0.8:
            status = "under"
        elif util <= 1.1:
            status = "near"
        else:
            status = "over"

        budget_view.append(
            {
                "category": b.category,
                "budget": round(budget_amount, 2),
                "actual": round(actual, 2),
                "utilisation": round(util, 2),
                "status": status,
            }
        )

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Money Copilot – Dashboard",
            "no_data": False,
            "no_period_data": False,
            "period": period_label,
            "available_periods": available_periods,
            "user_list": users,
            "selected_user_id": selected_user.id,
            "stats": stats,
            "top_categories": categories,
            "patterns": patterns,
            "story": story,
            "persona_label": persona_label,
            "persona_description": persona_description,
            "actions": actions,
            "category_trends": category_trends,
            "budget_view": budget_view,
        },
    )
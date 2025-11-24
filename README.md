# Personal Finance Copilot

A local-first, AI-assisted personal finance dashboard for young professionals.

This project ingests mock bank/UPI/credit card data, auto-categorises spending, detects “bad patterns”, and generates a plain-language **money story** for each month – plus concrete actions to try next month.

Built with **FastAPI**, **SQLite**, **Pandas**, **scikit-learn**, **Jinja2**, and the **OpenAI API**.

---

## 🎯 Problem & Concept

Young professionals (20s–30s) often:

- Don’t know where their money is really going
- Struggle to know how much they can safely spend
- Don’t have the time/energy to read complex statements or dashboards

This copilot:

- Connects to mock transaction data (simulating bank/UPI/credit card exports)
- Automatically categorises spending and finds patterns
- Tells the user a **simple story of their month in money**, with **concrete, non-judgmental suggestions**.

---

## ✨ Key Features

### 1. Data ingestion

- Import transactions via CSV (`/transactions/import-csv`)
- Supports multiple users (via `user_id`):
  - Simulate different financial personas (Saver, Spender, Subscription-heavy, etc.)
- Normalises:
  - `timestamp`
  - `amount`
  - `is_income`
  - Optional: `category`, `description`, `source`, `account_name`
- Ignores extra columns like `currency` safely

---

### 2. Analytics layer

#### Core metrics

For each month/user:

- Total income
- Total expense
- Net (income – expense)
- Savings rate

#### Auto-categorisation

Keyword-based, India-friendly categories such as:

- Food
- Shopping
- Transport
- Subscriptions
- Bills / Utilities
- Rent
- Other

(Implemented in `app/services/categorizer.py` – easy to extend.)

#### Pattern detection

From raw transactions, the analyzer detects:

- **High fees & charges**
  - Late fees, penalties, interest, convenience charges
- **Impulse spikes**
  - Days where spending is significantly above median daily expense
- **Subscription bloat**
  - Total subscription spend and transaction count
- **Cashflow health flag**
  - `ok` / `warning` / `critical` based on net and savings rate

(All implemented in `app/services/analyzer.py`.)

---

### 3. Behaviour profiles (clustering across months)

Across multiple months for a user, the project computes **behaviour profiles**:

- **Super Saver** – high savings rate, controlled expenses  
- **Balanced** – decent savings and spending under control  
- **Subscription Heavy** – significant recurring spend share  
- **Lifestyle Spender** – higher shopping / lifestyle share  

Uses:

- Monthly aggregates (savings rate, total expense)
- Subscription share of spend
- Shopping share of spend
- KMeans clustering (with safe fallback for low month counts)

Returned as:

- `labels_by_period` → `"2025-01": "Super Saver"`
- `cluster_descriptions` → explanation for each profile

---

### 4. Budget / target system

A very light budgeting layer:

- `Budget` table per user:
  - `category`
  - `amount` (monthly budget)
- Default budgets seeded (e.g., Food, Shopping, Subscriptions, Transport)

Dashboard shows:

- Actual vs budget per category for the selected month
- Status badge:
  - **Under budget**
  - **On track**
  - **Over budget**

This gives context to the analytics: not just “how much you spent”, but “relative to what you wanted”.

---

### 5. AI layer – Money story & actions

#### Money story

- `build_money_story` (in `app/services/storyteller.py`) calls OpenAI to generate:
  - A friendly, non-judgmental explanation of:
    - Income, expenses, net, savings rate
    - Top spending categories
    - Spikes, subscriptions, fees
  - 2–4 **concrete, simple suggestions** for next month

- If the API fails or is unavailable, there is a **deterministic fallback template** so the endpoint always returns something usable.

#### Action recommendations

- `recommend_actions_for_next_month` (in `app/services/actions.py`) builds a **short action list** based on:
  - Savings rate
  - Fees and charges
  - Impulse spike days and extra spend
  - Subscription total
  - Top categories

Examples:

- “Set up an automatic transfer of at least 10% of your income…”
- “Do a 10-minute subscription audit…”
- “Pick one of the high-spend days and plan in advance how you’ll handle a similar day next month…”

---

### 6. Dashboard UI

The main UI is a server-rendered dashboard at:

> `GET /dashboard`

Features:

- **User selector** (multi-user)
- **Month selector** (year + month)
- KPI tiles:
  - Month
  - Income, Expense, Net, Savings rate
  - Cashflow badge (OK/Warning/Critical)
  - Behaviour profile chip + description
- **Top spending categories** (for selected month)
- **Money story** (LLM-generated plain-language narrative)
- **Patterns we noticed**
  - Fees, impulse spikes, subscription summary
- **What to try next month** (action list)
- **Category trends vs typical month**
  - Current month vs average monthly spend in previous months
  - Sorted by biggest movers
- **Budgets vs actual**
  - For each budgeted category:
    - Actual / Budget
    - Under / On track / Over badges

The dashboard is fully server-rendered via **Jinja2**, so no frontend build step is required.

---

## 🧱 Architecture

**High-level components:**

- `app/main.py`  
  FastAPI app, routes, dependency injection, dashboard views

- `app/models.py`  
  SQLAlchemy models:
  - `User`
  - `Transaction`
  - `Budget`

- `app/database.py`  
  SQLite engine + session factory

- `app/services/categorizer.py`  
  Keyword-based auto-categorisation

- `app/services/analyzer.py`  
  - `compute_basic_stats`
  - `compute_category_breakdown`
  - `detect_patterns`
  - `build_behaviour_profiles`
  - `compute_category_trends`

- `app/services/storyteller.py`  
  - LLM prompt construction
  - OpenAI API call
  - Fallback template

- `app/services/actions.py`  
  - Generates concrete actions for next month

- `app/templates/`  
  - `index.html` – simple landing
  - `dashboard.html` – main UI

---

## 🗄️ Data model

### User

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=True)

    transactions = relationship("Transaction", back_populates="user")
    budgets = relationship("Budget", back_populates="user")
```

### Transaction

```python
class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    is_income = Column(Boolean, default=False, nullable=False)
    category = Column(String, nullable=True)
    description = Column(String, nullable=True)
    source = Column(String, nullable=True)
    account_name = Column(String, nullable=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="transactions")
```

### Budget

```python
class Budget(Base):
    __tablename__ = "budgets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    category = Column(String, nullable=False)
    amount = Column(Float, nullable=False)

    user = relationship("User", back_populates="budgets")
```

---

## 🔌 API Overview

Key endpoints:

- `GET /`  
  Simple landing page.

- `GET /dashboard`  
  Main dashboard (HTML).  
  Query params:
  - `user_id` (optional, defaults to first user)
  - `year` (optional)
  - `month` (optional)

- `POST /transactions/import-csv`  
  Import transactions from CSV.  
  Query params:
  - `user_id` (default: `1`)  
  Body:
  - `file` (CSV upload)

- `GET /analysis/monthly-story`  
  JSON money story for a given month.  
  Query params:
  - `user_id`
  - `year`
  - `month`

- `GET /analysis/actions-next-month`  
  JSON action list for what to change next month.  
  Query params:
  - `user_id`
  - `year`
  - `month`

(Exact params can be explored via the interactive docs.)

---

## ⚙️ Setup & Running Locally

### 1. Clone and create environment

```bash
git clone <this-repo-url>
cd money_copilot

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If you use `pyenv`, you can set the local version (e.g. 3.10.x):

```bash
pyenv local 3.10.11
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-key-here
```

> `.env` is included in `.gitignore` so your key never gets committed.

### 3. Run the app

```bash
uvicorn app.main:app --reload
```

- API docs: `http://127.0.0.1:8000/docs`
- Dashboard: `http://127.0.0.1:8000/dashboard`

### 4. Import mock data

Use the interactive docs:

1. Open `http://127.0.0.1:8000/docs`
2. Go to `POST /transactions/import-csv`
3. Upload `data/mock_transactions.csv`
4. (Optional) Choose `user_id` (default is `1`)

Now refresh `http://127.0.0.1:8000/dashboard`.

---

## 🧪 Example flows (for demo / portfolio)

You can demonstrate the product using flows like:

1. **First-time setup**
   - Import mock transactions for `Demo User`
   - Open dashboard → see first behaviour profile + story

2. **Show behaviour profile**
   - Use multiple months of data
   - Show how the label changes (e.g., from Balanced to Super Saver) as budgets and spend patterns vary

3. **Highlight subscriptions / fees**
   - Filter the CSV to add more subscription lines
   - Re-import for another `user_id`
   - Compare dashboards between users

4. **Budgets vs actual**
   - Adjust default budgets in the DB (or via a future endpoint)
   - Show under/on-track/over budget categories in the dashboard

---

## 🛠️ Extensibility Ideas

Future enhancements you (or recruiters) can imagine:

- Real bank / UPI / credit card connectors (Plaid-like)
- More sophisticated categorisation using ML
- Goal tracking (emergency fund, travel fund, debt payoff)
- Notifications / nudges via email or WhatsApp
- Mobile-friendly / React frontend

---

## 📄 License

MIT (or any license you prefer).

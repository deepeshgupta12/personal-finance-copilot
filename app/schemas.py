from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TransactionBase(BaseModel):
    timestamp: datetime
    amount: float
    currency: str = "INR"
    description: str

    source: Optional[str] = None
    account_name: Optional[str] = None
    is_income: bool = False
    category: Optional[str] = None  # will be filled by AI later


class TransactionCreate(TransactionBase):
    """Schema used when creating a transaction via API."""
    pass


class Transaction(TransactionBase):
    """Schema returned from API (includes DB id)."""
    id: int

    class Config:
        orm_mode = True

class MonthlySummary(BaseModel):
    period: str  # e.g. "2025-01"
    total_income: float
    total_expense: float
    net: float  # income - expense


class WeeklySummary(BaseModel):
    period: str  # e.g. "2025-W01"
    total_income: float
    total_expense: float
    net: float
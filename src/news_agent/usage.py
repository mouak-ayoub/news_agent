from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path

from .config import AppConfig
from .model import UsageStats


class BudgetExceededError(RuntimeError):
    pass


@dataclass(slots=True)
class CostReport:
    estimated_cost_usd: float
    monthly_total_usd: float


@dataclass(slots=True)
class UsageLedger:
    month: str
    total_cost_usd: float
    runs: list[dict]


class UsageGuard:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ledger_path = config.config_path.parents[1] / config.budget.ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, label: str, usage: UsageStats) -> CostReport:
        cost = self._estimate_cost(usage)
        if cost > self.config.budget.max_run_spend_usd:
            raise BudgetExceededError(
                f"Estimated run cost ${cost:.4f} exceeds max_run_spend_usd."
            )

        ledger = self._load_ledger()
        next_total = ledger.total_cost_usd + cost
        if next_total > self.config.budget.max_monthly_spend_usd:
            raise BudgetExceededError(
                "Estimated monthly spend would exceed max_monthly_spend_usd."
            )

        ledger.total_cost_usd = next_total
        ledger.runs.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "label": label,
                "cost_usd": cost,
                "usage": asdict(usage),
            }
        )
        self._save_ledger(ledger)
        return CostReport(estimated_cost_usd=cost, monthly_total_usd=next_total)

    def _estimate_cost(self, usage: UsageStats) -> float:
        input_cost = (
            usage.input_tokens / 1_000_000
        ) * self.config.budget.input_cost_per_million
        output_cost = (
            usage.output_tokens / 1_000_000
        ) * self.config.budget.output_cost_per_million
        search_cost = usage.web_search_calls * self.config.budget.web_search_cost_per_call
        return round(input_cost + output_cost + search_cost, 6)

    def _load_ledger(self) -> UsageLedger:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        if not self.ledger_path.exists():
            return UsageLedger(month=month, total_cost_usd=0.0, runs=[])

        data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if data.get("month") != month:
            return UsageLedger(month=month, total_cost_usd=0.0, runs=[])
        return UsageLedger(
            month=data["month"],
            total_cost_usd=float(data.get("total_cost_usd", 0.0)),
            runs=list(data.get("runs", [])),
        )

    def _save_ledger(self, ledger: UsageLedger) -> None:
        self.ledger_path.write_text(
            json.dumps(asdict(ledger), indent=2),
            encoding="utf-8",
        )

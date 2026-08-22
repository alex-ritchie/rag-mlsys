"""Demo-profile cost controls (spec §5.11): per-IP sliding-window rate limit + global daily spend cap."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from mlsys_common.models import Usage
from mlsys_common.settings import Settings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

RUN_IT_YOURSELF = "https://github.com/alexritchie/mlsysbook-rag#run-it-yourself"


class DemoLimitError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason  # rate_limit | budget
        self.message = message


@dataclass
class DemoGuard:
    engine: AsyncEngine
    settings: Settings
    salt: str = os.environ.get("DEMO_IP_SALT", "mlsysbook-demo")

    def ip_hash(self, ip: str) -> str:
        return hashlib.sha256(f"{self.salt}:{ip}".encode()).hexdigest()[:32]

    def cost_usd(self, usage: Usage) -> float:
        s = self.settings
        return (
            usage.prompt_tokens / 1e6 * s.demo_haiku_input_usd_per_mtok
            + usage.completion_tokens / 1e6 * s.demo_haiku_output_usd_per_mtok
        )

    async def spend_24h(self) -> float:
        async with self.engine.connect() as conn:
            v = (
                await conn.execute(
                    text(
                        "SELECT COALESCE(sum(cost_usd), 0) FROM demo_requests WHERE at > now() - interval '24 hours'"
                    )
                )
            ).scalar_one()
        return float(v)

    async def check(self, ip: str) -> None:
        """Raise DemoLimitError if this IP or the global budget is exhausted."""
        s = self.settings
        h = self.ip_hash(ip)
        async with self.engine.connect() as conn:
            n = (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM demo_requests WHERE ip_hash = :h AND at > now() - interval '24 hours'"
                    ),
                    {"h": h},
                )
            ).scalar_one()
            spent = (
                await conn.execute(
                    text(
                        "SELECT COALESCE(sum(cost_usd), 0) FROM demo_requests WHERE at > now() - interval '24 hours'"
                    )
                )
            ).scalar_one()
        if float(spent) >= s.demo_daily_budget_usd:
            raise DemoLimitError(
                "budget",
                f"The demo budget is exhausted for today. Run the full stack yourself: {RUN_IT_YOURSELF}",
            )
        if int(n) >= s.demo_rate_limit_per_day:
            raise DemoLimitError(
                "rate_limit",
                f"You've reached the demo limit of {s.demo_rate_limit_per_day} questions per day. Run it yourself: {RUN_IT_YOURSELF}",
            )

    async def record(self, ip: str, usage: Usage) -> float:
        cost = self.cost_usd(usage)
        async with self.engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO demo_requests (ip_hash, cost_usd) VALUES (:h, :c)"),
                {"h": self.ip_hash(ip), "c": cost},
            )
        return cost

    async def reserve(self, ip: str) -> None:
        """Insert a zero-cost row up front so concurrent requests count against the limit immediately."""
        async with self.engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO demo_requests (ip_hash, cost_usd) VALUES (:h, 0)"),
                {"h": self.ip_hash(ip)},
            )

    async def settle(self, ip: str, usage: Usage) -> float:
        """Attach the real cost to the most recent reservation for this IP."""
        cost = self.cost_usd(usage)
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE demo_requests SET cost_usd = :c WHERE id = (SELECT id FROM demo_requests WHERE ip_hash = :h ORDER BY at DESC LIMIT 1)"
                ),
                {"h": self.ip_hash(ip), "c": cost},
            )
        return cost

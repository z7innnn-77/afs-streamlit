"""Funded synthetic institutions, not downloaded institutional flow data."""
from dataclasses import dataclass
import math
import random


GROUPS = ("外資", "投信", "自營商")
SLEEVES = ("外資", "投信", "自營商自行買賣", "自營商避險")
FOREIGN_INITIAL_CASH = 5_000_000_000
TRUST_INITIAL_CASH = 200_000_000
LARGE_CAP_WEIGHT_PCT = 1.0


def initial_allocations(n, price, cap_weight):
    bonus = min(cap_weight / 5, 1) if cap_weight >= LARGE_CAP_WEIGHT_PCT else 0
    foreign_shares = round(160 * n * (1 + bonus))
    return ((10 * foreign_shares, FOREIGN_INITIAL_CASH * (1 + bonus)),
            (50 * n, TRUST_INITIAL_CASH),
            (20 * n, 4 * n * price),
            (20 * n, 4 * n * price))


@dataclass(frozen=True)
class InstitutionConfig:
    enabled: bool = False
    activity: float = 1.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or isinstance(self.activity, bool) or not isinstance(self.activity, (int, float)) or not math.isfinite(self.activity) or not 0 <= self.activity <= 3:
            raise ValueError("Invalid institutional activity")


@dataclass
class InstitutionAccount:
    sleeve: str
    group: str
    cash: float
    shares: int
    initial_shares: int
    average_cost: float


class InstitutionBook:
    def __init__(self, n, seed, initial_price, cap_weight=0, *, ownership=None):
        self.rng = random.Random(seed + 8000003)
        self.accounts = []
        allocations = initial_allocations(n, initial_price, cap_weight)
        if ownership is not None:
            foreign_pct, trust_pct = ownership
            total = 3500 * n
            foreign, trust, *dealers = allocations
            allocations = ((round(total * foreign_pct / 100), foreign[1]),
                           (round(total * trust_pct / 100), trust[1]), *dealers)
            self.retail_initial_shares = total - sum(shares for shares, _ in allocations)
        for sleeve, group, (shares, cash) in zip(SLEEVES, ("外資", "投信", "自營商", "自營商"), allocations):
            self.accounts.append(InstitutionAccount(sleeve, group, cash, shares, shares, initial_price))
        self.initial_cash = dict(zip(SLEEVES, (cash for _, cash in allocations)))
        self.n = n
        self.history = []
        self.day = None
        self.current = {}
        self.execution_plans = {}

    def upgrade_initial_capital(self, initial_price, cap_weight=0):
        if hasattr(self, "initial_cash"):
            return
        self.initial_cash = {}
        for account, (target_shares, target_cash) in zip(
                self.accounts, initial_allocations(self.n, initial_price, cap_weight)):
            original_cash = 2 * account.initial_shares * initial_price
            added_shares = max(0, target_shares - account.initial_shares)
            if added_shares:
                account.average_cost = (account.shares * account.average_cost + added_shares * initial_price) / (account.shares + added_shares)
                account.shares += added_shares
                account.initial_shares += added_shares
            account.cash = round(max(0, account.cash + target_cash - original_cash), 2)
            self.initial_cash[account.sleeve] = target_cash
            if self.current.get(account.sleeve, {}).get("狀態") == "盤中":
                self.current[account.sleeve].update({"持股": account.shares, "現金": account.cash,
                                                     "均價": account.average_cost})

    def begin_day(self, day):
        if day == self.day:
            return
        self.day = day
        self.current = {}
        for account in self.accounts:
            row = {"交易日": day, "法人": account.group, "分類": account.sleeve, "狀態": "盤中",
                   "買進股數": 0, "賣出股數": 0, "買賣超股數": 0,
                   "買進金額": 0.0, "賣出金額": 0.0, "買賣超金額": 0.0,
                   "持股": account.shares, "現金": account.cash, "均價": account.average_cost,
                   "策略": "尚未下單"}
            self.history.append(row)
            self.current[account.sleeve] = row

    def propose(self, config, price, upper, fundamental, momentum, trend, news, macro, quote_tick, cap_weight,
                *, execution=None):
        noise = [self.rng.gauss(0, 0.012) for _ in SLEEVES]
        if not config.enabled or config.activity == 0:
            return []
        value = max(-0.3, min(0.3, fundamental / price - 1))
        plans = []
        for i, account in enumerate(self.accounts):
            if i == 1 and quote_tick % 3:
                continue
            wealth = account.cash + account.shares * price
            initial_weight = account.initial_shares * price / (
                self.initial_cash[account.sleeve] + account.initial_shares * price)
            if i == 0:
                target = wealth * max(0, min(0.95, initial_weight + min(cap_weight / 10, 1) * 0.1 +
                                             0.7 * value + 2 * macro + 0.25 * trend + 0.035 * news + noise[i])) / price
                reason = "權值偏好／估值與市場氣氛"
            elif i == 1:
                target = wealth * max(0, min(0.95, initial_weight + 0.5 * value + 0.8 * trend + 0.04 * news + noise[i])) / price
                reason = "中期趨勢／估值配置"
            elif i == 2:
                target = wealth * max(0, min(0.95, 0.34 + 1.3 * momentum + 0.5 * value + noise[i])) / price
                reason = "短線動能／價差交易"
            else:
                target = account.initial_shares * max(0.5, min(1.5, 1 - 3 * momentum - trend + noise[i]))
                reason = "庫存風險調整（非衍生品避險）"
            desired = round(target) - account.shares
            if execution is None:
                maximum = max(1, int(self.n * (1.2 if i == 0 else 0.08 if i == 1 else 0.06) * config.activity))
            else:
                depth, prior_volume, lower, ticks_per_day = execution
                participation = (0.32, 0.12, 0.06, 0.06)[i] * config.activity
                maximum = min(max(1, int(depth * min(0.6, participation))),
                              max(1, int(prior_volume * min(0.35, participation))))
                if i < 2:
                    session = quote_tick // ticks_per_day
                    plan = self.execution_plans.get(account.sleeve)
                    if plan is None or session >= plan["end"]:
                        days = 4 if i == 0 else 5
                        intervals = days * ticks_per_day // (1 if i == 0 else 3)
                        total = max(-maximum * intervals, min(maximum * intervals, desired))
                        plan = {"start": session, "end": session + days, "total": total, "remaining": total,
                                "reason": "外資分批調整權值曝險" if i == 0 else "投信分批再平衡"}
                        self.execution_plans[account.sleeve] = plan
                    desired = plan["remaining"]
                    remaining_ticks = plan["end"] * ticks_per_day - quote_tick
                    slices = max(1, math.ceil(remaining_ticks / (1 if i == 0 else 3)))
                    maximum = min(maximum, math.ceil(abs(desired) / slices))
                    elapsed = session - plan["start"] + 1
                    duration = plan["end"] - plan["start"]
                    reason = f'{plan["reason"]}｜第 {elapsed}/{duration} 日'
                    row = self.current[account.sleeve]
                    row.update({"策略": reason, "計畫目標股數": plan["total"], "計畫剩餘股數": desired})
                if (desired > 0 and price >= upper) or (desired < 0 and price <= lower):
                    maximum = 0
            quantity = max(-maximum, min(maximum, desired))
            quantity = min(quantity, int(account.cash // upper)) if quantity > 0 else max(quantity, -account.shares)
            if quantity:
                plans.append((account, quantity, reason))
        return plans

    def settle(self, plans, price):
        for account, quantity, reason in plans:
            amount = round(abs(quantity) * price, 2)
            if quantity > 0:
                account.average_cost = (account.average_cost * account.shares + amount) / (account.shares + quantity)
            account.cash = round(account.cash - quantity * price, 2)
            account.shares += quantity
            row = self.current[account.sleeve]
            if quantity > 0:
                row["買進股數"] += quantity
                row["買進金額"] += amount
            else:
                row["賣出股數"] -= quantity
                row["賣出金額"] += amount
            row["買賣超股數"] = row["買進股數"] - row["賣出股數"]
            row["買賣超金額"] = round(row["買進金額"] - row["賣出金額"], 2)
            row["策略"] = reason
            if "計畫剩餘股數" in row:
                plan = self.execution_plans[account.sleeve]
                plan["remaining"] -= quantity
                row["計畫剩餘股數"] = plan["remaining"]
        for account in self.accounts:
            self.current[account.sleeve].update({"持股": account.shares, "現金": account.cash,
                                                   "均價": account.average_cost})

    def close_day(self):
        for row in self.current.values():
            row["狀態"] = "收盤"

    def grouped_history(self):
        grouped = {}
        for row in self.history:
            key = row["交易日"], row["法人"]
            if key not in grouped:
                grouped[key] = {"交易日": key[0], "法人": key[1], "狀態": row["狀態"],
                                **{name: 0 for name in ("買進股數", "賣出股數", "買賣超股數", "買進金額", "賣出金額", "買賣超金額", "持股", "現金")}}
            for name in ("買進股數", "賣出股數", "買賣超股數", "買進金額", "賣出金額", "買賣超金額", "持股", "現金"):
                grouped[key][name] += row[name]
        return list(grouped.values())

    def snapshot(self, other_shares=0):
        denominator = other_shares + sum(account.shares for account in self.accounts)
        history = self.grouped_history()
        rows = []
        for group in GROUPS:
            today = next((row for row in reversed(history) if row["交易日"] == self.day and row["法人"] == group), None)
            closed = [row for row in history if row["法人"] == group and row["狀態"] == "收盤"]
            streak = 0
            sign = 1 if closed and closed[-1]["買賣超股數"] > 0 else -1 if closed and closed[-1]["買賣超股數"] < 0 else 0
            for row in reversed(closed):
                if not sign or row["買賣超股數"] * sign <= 0:
                    break
                streak += sign
            held = sum(account.shares for account in self.accounts if account.group == group)
            rows.append({"法人": group, "買進股數": today["買進股數"] if today else 0,
                         "賣出股數": today["賣出股數"] if today else 0,
                         "買賣超股數": today["買賣超股數"] if today else 0,
                         "買進金額": today["買進金額"] if today else 0,
                         "賣出金額": today["賣出金額"] if today else 0,
                         "買賣超金額": today["買賣超金額"] if today else 0,
                         "持股": held, "參與者持股占比 (%)": held / max(1, denominator) * 100,
                         "連買賣日數（收盤）": streak,
                         "近5收盤日買賣超": sum(row["買賣超股數"] for row in closed[-5:]),
                         "近20收盤日買賣超": sum(row["買賣超股數"] for row in closed[-20:])})
        return rows

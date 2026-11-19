"""Synthetic five-level liquidity and paper-account transaction costs."""
import json
import math
from datetime import date
from pathlib import Path

CALIBRATION = json.loads(Path(__file__).with_name("data").joinpath("market_calibration.json").read_text(encoding="utf-8"))
BROKER_FEE_RATE = 0.001425
DAY_TRADE_TAX_END = date(2027, 12, 31).isoformat()


def fee(amount):
    return round(amount * BROKER_FEE_RATE, 2)


def sales_tax(price, quantity, is_etf=False, daytrade_quantity=0):
    rate = 0.001 if is_etf else 0.003
    return round(price * (quantity * rate - daytrade_quantity * 0.0015), 2)


def tick_cents(cents, is_etf):
    if is_etf:
        return 1 if cents < 5000 else 5
    if cents < 1000:
        return 1
    if cents < 5000:
        return 5
    if cents < 10000:
        return 10
    if cents < 50000:
        return 50
    if cents < 100000:
        return 100
    return 500


def align_price(price, is_etf, *, upward):
    cents = round(price * 100)
    tick = tick_cents(cents, is_etf)
    aligned = cents + (-cents % tick) if upward else cents - cents % tick
    return aligned / 100


def liquidity_size(profile, n, fallback):
    if not profile:
        return max(1, round(n * fallback))
    # Rescale empirical volumes to the simulation population.
    return max(1, round(n * (profile["median_volume"] / 5_000_000) ** 0.45))


def quote_volatility(profile, ticks):
    # High/low range provides a within-session scale proxy without tick data.
    typical_range = profile["median_range"]
    daily_sigma = profile["daily_volatility"]
    return min(0.06, max(0.004, (typical_range / 1.6 + daily_sigma) / 2)) / math.sqrt(ticks)


def build_book(reference, lower, upper, is_etf, depth, session_phase, ticks):
    center = round(reference * 100)
    lower, upper = round(lower * 100), round(upper * 100)
    # Stock depth is generally greater near the beginning and end of the day.
    activity = 0.75 + 0.6 * abs(2 * session_phase / (ticks - 1) - 1)
    book = {"買入": [], "賣出": []}
    for side, direction in (("買入", -1), ("賣出", 1)):
        price = center
        for level in range(5):
            if direction < 0:
                price -= 1
                price -= price % tick_cents(price, is_etf)
            else:
                price += 1
                step = tick_cents(price, is_etf)
                price += (-price) % step
            if not lower <= price <= upper:
                break
            quantity = max(1, round(depth * activity * (0.6 + level * 0.2)))
            book[side].append([price / 100, quantity])
    return book


def sweep_levels(levels, quantity, limit=None, *, buying=True):
    for level in levels:
        price, available = level
        if limit is not None and (price > limit if buying else price < limit):
            break
        filled = min(quantity, available)
        if filled:
            yield level, filled
            quantity -= filled
        if not quantity:
            break


def preview_sweep(levels, quantity):
    fills = [(level[0], filled) for level, filled in sweep_levels(levels, quantity)]
    return sum(q for _, q in fills), round(sum(p * q for p, q in fills), 2)


def match_player(book, market_orders, limit_orders):
    fills = []
    for side, opposite in (("買入", "賣出"), ("賣出", "買入")):
        market = sorted((o for o in market_orders if o.side == side), key=lambda o: o.ident)
        limits = sorted((o for o in limit_orders if o.side == side),
                        key=lambda o: (-o.limit_price if side == "買入" else o.limit_price, o.ident))
        for order in market + limits:
            for level, quantity in sweep_levels(book[opposite], order.remaining, order.limit_price,
                                                buying=side == "買入"):
                level[1] -= quantity
                fills.append((order, level[0], quantity))
    return fills

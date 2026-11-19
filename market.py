"""AFS: seeded daily sessions, simulated quotes and personal paper trading.

A simplified market maker absorbs net order flow. This is NOT a real exchange
order-book/matching engine, nor does it predict real-world share prices.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import defaultdict, deque
from bisect import bisect_right
from pathlib import Path
import json
from datetime import date, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math
import random
import zlib
from typing import Optional

from research_models import ModelConfig, ResearchEngine, Observation
from execution import (CALIBRATION, DAY_TRADE_TAX_END, align_price, build_book, fee, liquidity_size,
                       match_player, quote_volatility, sales_tax, tick_cents)
from market_news import (AutoNewsConfig, EXTREME_SCENARIOS, EXTREME_SCENARIO_CHANCE, MARKET_SCENARIOS,
                         ORDINARY_SCENARIOS, ORDINARY_SCENARIO_CHANCE, RATE_SENSITIVITY,
                         analyst_headline, draw_news, explain_move)
from institutions import InstitutionConfig, InstitutionBook, GROUPS, LARGE_CAP_WEIGHT_PCT

STYLES = {
    "動能型": {"trend": 1.4, "news": 0.65, "value": 0.10, "herd": 0.30},
    "價值型": {"trend": -0.10, "news": 0.15, "value": 1.65, "herd": -0.05},
    "恐慌型": {"trend": 0.75, "news": 1.25, "value": 0.10, "herd": 0.60},
    "逆向型": {"trend": -1.10, "news": -0.20, "value": 0.95, "herd": -0.35},
    "消息型": {"trend": 0.25, "news": 1.65, "value": 0.05, "herd": 0.05},
    "從眾型": {"trend": 0.35, "news": 0.55, "value": 0.05, "herd": 1.45},
    "懷疑型": {"trend": 0.20, "news": 0.30, "value": 0.80, "herd": -0.10},
    "隨機型": {"trend": 0.05, "news": 0.10, "value": 0.05, "herd": 0.05},
}


@dataclass(frozen=True)
class Stock:
    symbol: str
    name: str
    sector: str
    initial_price: float
    volatility: float
    liquidity: float
    code: str = ""
    exchange: str = ""
    tag: str = "一般"
    day_trade_verified: bool = False
    day_trade_asof: str = ""
    taiwan_weight_pct: float = 0.0
    day_trade_source: str = ""
    active: bool = True
    basket: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ETFAllocation:
    day: date
    units: tuple[tuple[str, float], ...]


CATALOG = json.loads((Path(__file__).resolve().parent / "data" / "stock_catalog.json").read_text(encoding="utf-8"))
MARKET_CLOSE = json.loads((Path(__file__).resolve().parent / "data" / "market_close_20261102.json").read_text(encoding="utf-8"))
FOREIGN_OWNERSHIP = json.loads((Path(__file__).resolve().parent / "data" / "foreign_ownership_20261102.json").read_text(encoding="utf-8"))
STOCKS = tuple(Stock(**({key: row[key] for key in Stock.__dataclass_fields__ if key in row}
                        | {"initial_price": MARKET_CLOSE["close"][row["code"]]}))
               for row in CATALOG["stocks"])
ACTIVE_ETF_TAG = "主動式ETF"
TAIWAN_50_BASKET = (("AF01", .55), ("AF13", .15), ("AF14", .12), ("AF15", .10), ("AF18", .08))
ETFS = (
    Stock("AF54", "元大台灣50", "ETF", MARKET_CLOSE["close"]["0050"], 0.35, 3, code="0050", exchange="上市",
          basket=TAIWAN_50_BASKET),
    Stock("AF55", "元大高股息", "ETF", MARKET_CLOSE["close"]["0056"], 0.3, 3, code="0056", exchange="上市",
          basket=(("AF07", .25), ("AF08", .20), ("AF19", .20), ("AF26", .20), ("AF50", .15))),
    Stock("AF56", "國泰永續高股息", "ETF", MARKET_CLOSE["close"]["00878"], 0.3, 3, code="00878", exchange="上市",
          basket=(("AF07", .25), ("AF21", .25), ("AF22", .20), ("AF19", .15), ("AF17", .15))),
    Stock("AF57", "群益台灣精選高息", "ETF", MARKET_CLOSE["close"]["00919"], 0.35, 3, code="00919", exchange="上市",
          basket=(("AF07", .25), ("AF20", .20), ("AF23", .20), ("AF50", .20), ("AF51", .15))),
)
ADDED_ETFS = (
    ("AF78", "00980A", "主動野村臺灣優選", ACTIVE_ETF_TAG,
     (("AF01", .22), ("AF15", .23), ("AF37", .20), ("AF16", .20), ("AF07", .15))),
    ("AF79", "00981A", "主動統一台股增長", ACTIVE_ETF_TAG,
     (("AF01", .25), ("AF02", .22), ("AF41", .20), ("AF42", .18), ("AF18", .15))),
    ("AF80", "00982A", "主動群益台灣強棒", ACTIVE_ETF_TAG,
     (("AF14", .23), ("AF37", .22), ("AF03", .20), ("AF04", .20), ("AF70", .15))),
    ("AF81", "00984A", "主動安聯台灣高息", ACTIVE_ETF_TAG,
     (("AF07", .25), ("AF21", .20), ("AF19", .20), ("AF27", .20), ("AF50", .15))),
    ("AF82", "00985A", "主動野村台灣50", ACTIVE_ETF_TAG,
     (("AF01", .42), ("AF13", .18), ("AF14", .16), ("AF15", .14), ("AF20", .10))),
    ("AF83", "00987A", "主動台新優勢成長", ACTIVE_ETF_TAG,
     (("AF01", .24), ("AF41", .22), ("AF36", .19), ("AF37", .18), ("AF68", .17))),
    ("AF84", "00991A", "主動復華未來50", ACTIVE_ETF_TAG,
     (("AF01", .28), ("AF42", .22), ("AF04", .18), ("AF71", .17), ("AF66", .15))),
    ("AF85", "00992A", "主動群益科技創新", ACTIVE_ETF_TAG,
     (("AF01", .24), ("AF13", .21), ("AF42", .20), ("AF43", .18), ("AF72", .17))),
    ("AF86", "006208", "富邦台50", "一般ETF", TAIWAN_50_BASKET),
    ("AF87", "00692", "富邦公司治理", "一般ETF",
     (("AF01", .45), ("AF14", .17), ("AF15", .15), ("AF19", .13), ("AF07", .10))),
    ("AF88", "00713", "元大台灣高息低波", "一般ETF",
     (("AF19", .24), ("AF26", .22), ("AF11", .20), ("AF08", .18), ("AF22", .16))),
    ("AF89", "00881", "國泰台灣科技龍頭", "一般ETF",
     (("AF01", .40), ("AF13", .20), ("AF14", .16), ("AF15", .14), ("AF18", .10))),
    ("AF90", "00892", "富邦台灣半導體", "一般ETF",
     (("AF01", .32), ("AF13", .22), ("AF16", .18), ("AF17", .16), ("AF43", .12))),
    ("AF91", "00915", "凱基優選高股息30", "一般ETF",
     (("AF07", .24), ("AF22", .21), ("AF21", .20), ("AF27", .18), ("AF52", .17))),
    ("AF92", "00923", "群益台ESG低碳50", "一般ETF",
     (("AF01", .42), ("AF13", .18), ("AF14", .16), ("AF19", .14), ("AF20", .10))),
    ("AF93", "00929", "復華台灣科技優息", "一般ETF",
     (("AF17", .23), ("AF03", .22), ("AF27", .20), ("AF23", .17), ("AF19", .18))),
)
ETFS += tuple(
    Stock(symbol, name, "ETF", MARKET_CLOSE["close"][code], 0.35, 3,
          code=code, exchange="上市", tag=tag, basket=basket)
    for symbol, code, name, tag, basket in ADDED_ETFS
)
NEW_ETF_SYMBOLS = {symbol for symbol, *_ in ADDED_ETFS}
INDUSTRY_EXPANSION = frozenset(f"AF{ident}" for ident in range(94, 109))
INSTRUMENTS = STOCKS + ETFS


AI_INITIAL_SHARES = 20
RETAIL_INITIAL_CASH = 2_000_000


@dataclass
class Agent:
    ident: int
    style: str
    trust: float
    verify: float
    cash: float = 10000.0
    shares: int = AI_INITIAL_SHARES
    fear: float = 0.2
    action: str = "尚未交易"
    reason: str = "等待第一回合"
    last_qty: int = 0
    estimate: float = 0.0
    cost_basis: float = 0.0
    peak_wealth: float = 0.0
    belief: float = 0.0
    forecast: float = 0.0
    forecast_error: Optional[float] = None
    personality: str = "未啟用"
    model_family: str = "原始規則基準"

    def wealth(self, price: float) -> float:
        return self.cash + self.shares * price


@dataclass
class Event:
    title: str
    tone: int
    power: float
    reach: float
    truth: str
    anchor: str
    at_tick: int
    exposed: set[int]
    source: str = "手動"
    published_at: str = ""
    scope: str = ""


QUOTE_MINUTES = 5
TICKS_PER_DAY = 270 // QUOTE_MINUTES
MODEL_VERSION = 29
SCENARIO_DURATION_DAYS = 6
EXTREME_VOLATILITY_TAG = "極端波動"
ETF_TRACKING_BAND = 0.003
FOREIGN_PRICE_IMPACT_MULTIPLIER = 6
MAX_DEPOSIT = 1_000_000_000_000
INDEX_WEIGHTS = {row["symbol"]: row["index_weight_pct"] for row in CATALOG["stocks"]}
INDEX_WEIGHTS.update({stock.symbol: 0.0 for stock in ETFS})


def next_weekday(day):
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def price_limits(reference):
    """Round inward to cents so every print respects the daily 10% band."""
    value = Decimal(str(reference))
    lower = (value * Decimal("0.9")).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    upper = (value * Decimal("1.1")).quantize(Decimal("0.01"), rounding=ROUND_FLOOR)
    return max(0.01, float(lower)), max(0.01, float(upper))


@dataclass
class Position:
    quantity: int = 0
    cost: float = 0.0


@dataclass
class Order:
    ident: int
    symbol: str
    side: str
    quantity: int
    reserved_cash: float = 0.0
    status: str = "待成交"
    fill_price: Optional[float] = None
    submitted_at: str = ""
    filled_at: str = ""
    order_type: str = "市價"
    limit_price: Optional[float] = None
    valid_day: str = ""
    filled_quantity: int = 0
    filled_value: float = 0.0
    includes_fee: bool = False

    @property
    def remaining(self):
        return self.quantity - self.filled_quantity


class Player:
    def __init__(self, cash=1_000_000.0):
        self.initial_cash = float(cash)
        self.cash = float(cash)
        self.deposited_cash = 0.0
        self.deposits: list[tuple[str, int]] = []
        self.realized = 0.0
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.trades: list[dict] = []
        self.intraday_inventory = {}
        self.inventory_day = None

    @property
    def contributed_capital(self):
        return self.initial_cash + self.deposited_cash

    def deposit(self, amount: int, *, timestamp: str | None = None):
        if type(amount) is not int or not 1 <= amount <= MAX_DEPOSIT:
            raise ValueError(f"單次入金須為 1 至 {MAX_DEPOSIT:,} 元的整數。")
        self.cash = round(self.cash + amount, 2)
        self.deposited_cash += amount
        if timestamp is not None:
            if not hasattr(self, "deposits"):
                self.deposits = []
            self.deposits.append((timestamp, amount))

    @property
    def pending(self):
        return [order for order in self.orders if order.status in ("待成交", "部分成交")]

    @property
    def available_cash(self):
        return round(self.cash - sum(o.reserved_cash for o in self.pending), 2)

    def available_shares(self, symbol):
        held = self.positions.get(symbol, Position()).quantity
        return held - sum(o.remaining for o in self.pending if o.symbol == symbol and o.side == "賣出")

    def wealth(self, markets):
        return self.cash + sum(p.quantity * markets[s].price for s, p in self.positions.items())

    def daytrade_summary(self, day):
        queues = defaultdict(deque)
        quantity, spread, profit = 0, 0.0, 0.0
        for trade in self.trades:
            if not trade["成交時間"].startswith(day):
                continue
            symbol = trade["股票代碼"]
            if trade["買賣"] == "買入":
                queues[symbol].append([trade["股數"], trade["成交價"], trade.get("手續費", 0) / trade["股數"]])
                continue
            remaining = trade["股數"]
            while remaining and queues[symbol]:
                lot = queues[symbol][0]
                matched = min(remaining, lot[0])
                quantity += matched
                tax = trade.get("當沖證交稅", trade.get("證交稅", 0))
                tax_quantity = trade.get("當沖股數") or trade["股數"]
                spread += matched * (trade["成交價"] - lot[1])
                profit += matched * (trade["成交價"] - lot[1] - lot[2] - trade.get("手續費", 0) / trade["股數"]
                                     - tax / tax_quantity)
                lot[0] -= matched
                remaining -= matched
                if lot[0] == 0:
                    queues[symbol].popleft()
        return {"股數": quantity, "價差損益": round(spread, 2), "淨損益": round(profit, 2)}

    def fill(self, order, price, timestamp, name, quantity=None, *, stock=None):
        quantity = order.remaining if quantity is None else quantity
        gross = round(price * quantity, 2)
        commission = round(fee(order.filled_value + gross) - fee(order.filled_value), 2) if stock else 0
        tax = daytrade_tax = matched = 0
        position = self.positions.setdefault(order.symbol, Position())
        realized = 0.0
        if stock and self.inventory_day != timestamp[:10]:
            self.inventory_day = timestamp[:10]
            self.intraday_inventory.clear()
        same_day = self.intraday_inventory.get(order.symbol, 0)
        if order.side == "買入":
            self.cash = round(self.cash - gross - commission, 2)
            position.quantity += quantity
            position.cost += gross + commission
            if stock:
                self.intraday_inventory[order.symbol] = same_day + quantity
        else:
            if stock:
                matched = min(quantity, same_day)
                self.intraday_inventory[order.symbol] = same_day - matched
                eligible = matched if stock.day_trade_verified and not stock.basket and timestamp[:10] <= DAY_TRADE_TAX_END else 0
                tax = sales_tax(price, quantity, bool(stock.basket), eligible)
                daytrade_tax = sales_tax(price, matched, bool(stock.basket), eligible)
            cost = position.cost * quantity / position.quantity
            position.quantity -= quantity
            position.cost = position.cost - cost if position.quantity else 0.0
            realized = gross - cost - commission - tax
            self.realized += realized
            self.cash = round(self.cash + gross - commission - tax, 2)
        order.filled_quantity += quantity
        order.filled_value = round(order.filled_value + gross, 2)
        if stock and order.side == "買入":
            if order.limit_price is not None:
                unfilled_value = round(order.limit_price * order.remaining, 2)
                order.reserved_cash = round(unfilled_value + fee(order.filled_value + unfilled_value) -
                                            fee(order.filled_value), 2)
            else:
                order.reserved_cash = round(max(0, order.reserved_cash - gross - commission), 2)
        order.status = "部分成交" if order.remaining else "已成交"
        order.fill_price = price if order.filled_quantity == quantity else order.filled_value / order.filled_quantity
        order.filled_at = timestamp
        self.trades.append({
            "委託編號": order.ident, "成交時間": timestamp, "股票代碼": order.symbol,
            "股票名稱": name, "買賣": order.side, "股數": quantity,
            "買方": "玩家" if order.side == "買入" else "做市商",
            "賣方": "做市商" if order.side == "買入" else "玩家",
            "成交價": price, "成交金額": gross, "已實現損益": round(realized, 2),
            "手續費": commission, "證交稅": tax, "當沖證交稅": daytrade_tax, "當沖股數": matched,
            "交割金額": round(-gross - commission if order.side == "買入" else gross - commission - tax, 2),
            "委託類型": order.order_type, "委託價": order.limit_price,
        })


class Market:
    def __init__(self, n: int = 100, seed: int = 42, stock: Optional[Stock] = None,
                 volatility: float = 1.5, start_date: date = date(2026, 11, 2), model_config=None, institution_config=None):
        if n < 1:
            raise ValueError("n must be positive")
        if start_date.weekday() >= 5:
            raise ValueError("Start date must be a weekday")
        self.stock = stock or STOCKS[0]
        self.volatility = volatility
        self.realistic = False
        self.book = None
        self.seed, self.n = seed, n
        self.rng = random.Random(seed)
        self.news_rng = random.Random(seed + 1000003)
        self.environment_rng = random.Random(seed + 2000003)
        self.tick = self.quote_tick = self.phase = 0
        self.start_date = start_date
        self.trading_date = start_date - timedelta(days=1)
        while self.trading_date.weekday() >= 5:
            self.trading_date -= timedelta(days=1)
        self.price = float(self.stock.initial_price)
        self.previous_price = self.previous_close = self.price
        self.fundamental = self.price
        self.etf_allocation: Optional[ETFAllocation] = None
        self.sentiment = 0.0
        self.risk = 1.0
        self.theme_impulse = 0.0
        self.activity_rng = random.Random(seed + 11000003)
        self.quote_minutes = QUOTE_MINUTES
        self.previous_imbalance = 0.0
        self.event: Optional[Event] = None
        cash_per_agent, remainder = divmod(RETAIL_INITIAL_CASH, n)
        self.retail_initial_cash = RETAIL_INITIAL_CASH
        foreign_pct = FOREIGN_OWNERSHIP["foreign_pct"].get(self.stock.code)
        self.initial_ownership = None if foreign_pct is None else (foreign_pct, 3.0)
        self.institutions = InstitutionBook(n, seed, self.price, self.stock.taiwan_weight_pct,
                                            ownership=self.initial_ownership)
        self.retail_initial_shares = getattr(self.institutions, "retail_initial_shares", n * AI_INITIAL_SHARES)
        shares_per_agent, extra_shares = divmod(self.retail_initial_shares, n)
        self.agents = [
            Agent(ident=i + 1, style=list(STYLES)[i % len(STYLES)],
                  trust=self.rng.uniform(0.25, 0.95), verify=self.rng.uniform(0.1, 0.95),
                  cash=cash_per_agent + (i < remainder), shares=shares_per_agent + (i < extra_shares))
            for i in range(n)
        ]
        self.daily_agent_volume = [0] * n
        self.history = [self._record("起點")]
        self.quotes: list[dict] = []
        self.fill_batches = {}
        self.log_batches: list[bytes] = []
        self.events: list[dict] = []
        self.last_explanation = None
        self.model_config = model_config or ModelConfig.preset("legacy")
        self.research_engine = None
        self.return_variance = 0.0001
        for agent in self.agents:
            agent.cost_basis = self.price
            agent.peak_wealth = agent.wealth(self.price)
        self.configure_model(self.model_config)
        self.configure_institutions(institution_config or InstitutionConfig())

    def configure_institutions(self, config):
        if not isinstance(config, InstitutionConfig):
            # Streamlit can reload the settings module while this market stays alive.
            try:
                config = InstitutionConfig(enabled=config.enabled, activity=config.activity)
            except AttributeError as error:
                raise ValueError("Invalid institutional configuration") from error
        self.institution_config = replace(config, enabled=False) if self.model_config.mode == "zi" or not self.stock.active else config

    def institution_snapshot(self, player_shares=0):
        return self.institutions.snapshot(sum(a.shares for a in self.agents) + player_shares)

    def configure_model(self, config):
        if not isinstance(config, ModelConfig):
            raise ValueError("Invalid model configuration")
        self.model_config = config
        if config.mode == "zi" and hasattr(self, "institution_config"):
            self.institution_config = replace(self.institution_config, enabled=False)
        if config.mode != "legacy":
            if self.research_engine is None:
                self.research_engine = ResearchEngine(self.agents, self.seed, self.stock.initial_price, config)
            else:
                self.research_engine.configure(config)
        else:
            for agent in self.agents:
                agent.model_family = "原始規則基準"

    def _record(self, status):
        lower, upper = self.trading_limits(self.previous_close)
        return {
            "回合": self.tick + (1 if status == "盤中" else 0),
            "交易日": self.trading_date.isoformat(), "狀態": status,
            "昨收": self.previous_close, "開盤": self.price, "最高": self.price,
            "最低": self.price, "收盤": self.price, "股價": self.price,
            "漲跌幅 (%)": 0.0, "成交量": 0, "淨需求": 0,
            "買入人數": 0, "賣出人數": 0, "觀望人數": self.n,
            "跌停價": lower, "漲停價": upper, "市場衝擊 (%)": 0.0,
        }

    def trading_limits(self, reference):
        lower, upper = price_limits(reference)
        if self.realistic:
            is_etf = bool(self.stock.basket)
            lower = align_price(lower, is_etf, upward=True)
            upper = align_price(upper, is_etf, upward=False)
        return lower, upper

    @property
    def next_limits(self):
        return self.trading_limits(self.price if self.phase == 0 else self.previous_close)

    @property
    def clock(self):
        minutes = 9 * 60 + (self.phase if self.phase else TICKS_PER_DAY) * QUOTE_MINUTES
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    @property
    def calibration(self):
        return CALIBRATION["stocks"].get(self.stock.code)

    @property
    def depth(self):
        size = liquidity_size(self.calibration, self.n, self.stock.liquidity)
        return max(1, round(size * 0.60)) if self.stock.tag == EXTREME_VOLATILITY_TAG else size

    @property
    def thinly_traded(self):
        return bool(self.calibration and self.calibration["median_volume"] < 600_000)

    def participation(self, event):
        attention = 0.0
        if event and event.source == "模擬分析師":
            age = self.quote_tick - self.event_quote_tick
            if age < 36:
                attention = event.power * 0.5 ** (age / 9)
        volume_ratio = self.calibration["median_volume"] / 20_000_000 if self.calibration else 1.0
        roll = self.activity_rng.random()
        if self.thinly_traded and attention < 0.20 and roll < 0.68:
            return 0.0, attention
        session_profile = 0.65 + 0.35 * abs(2 * self.phase / (TICKS_PER_DAY - 1) - 1)
        activity = min(1.0, max(0.03, volume_ratio ** 0.36) * session_profile * (0.35 + 1.05 * roll))
        if attention:
            activity = max(activity, min(1.0, 1.2 * attention))
        return activity, attention

    def indicative_book(self):
        return build_book(self.price, *self.next_limits, bool(self.stock.basket), self.depth,
                          self.phase, TICKS_PER_DAY)

    def announce(self, title, tone, power, reach, truth, anchor, *, source="手動", published_at=None, scope="個股"):
        exposed = {a.ident for a in self.agents if self.news_rng.random() < reach / 100}
        timestamp = published_at or f"{self.trading_date.isoformat()} {self.clock}"
        self.event = Event(title.strip(), tone, power / 100, reach / 100, truth, anchor, self.tick, exposed,
                           source, timestamp, scope)
        self.event_quote_tick = self.quote_tick
        self.events.append({"回合": self.tick, "交易日": timestamp[:10], "發布時間": timestamp,
                            "來源": source, "影響範圍": scope,
                            "消息": title.strip(), "消息傾向": tone, "真相": truth,
                            "主播立場": anchor, "觸及人數": len(exposed)})

    def quote(self, common_shock=0.0, sector_shock=0.0, player_net=0, player_volume=0, limit_orders=(),
              *, market_orders=(), basket_value=None, basket_previous_close=None, rate_pressure=0.0, scenario_effect=None):
        if self.phase == 0 and self.history[-1]["狀態"] != "盤中":
            self.trading_date = self.start_date if self.tick == 0 else next_weekday(self.trading_date)
            self.previous_close = self.price
            self.history.append(self._record("盤中"))
            self.daily_agent_volume = [0] * self.n
        self.institutions.begin_day(self.trading_date.isoformat())
        lower, upper = self.trading_limits(self.previous_close)
        env = self.environment_rng
        scale = math.sqrt(TICKS_PER_DAY)
        self.fundamental = max(0.01, self.fundamental * math.exp(env.gauss(0, 0.006 / scale)))
        self.sentiment = 0.86 ** (1 / TICKS_PER_DAY) * self.sentiment + env.gauss(0, 0.18 / scale)
        self.risk = min(2.5, max(0.65, 0.99 * self.risk + 0.01 + env.gauss(0, 0.10 / scale)))
        local_shock = env.gauss(0, 0.014) * self.risk
        jump_roll, jump_size = env.random(), env.gauss(0, 0.065)
        jump = jump_size if jump_roll < 0.05 / TICKS_PER_DAY else 0.0
        shock = self.volatility * self.stock.volatility * (
            (common_shock + sector_shock + local_shock) / scale + jump)
        shock *= self.model_config.exogenous_scale
        factor = self.volatility * self.stock.volatility * self.model_config.exogenous_scale
        if self.realistic and self.calibration:
            factor = self.volatility / 1.5 * self.model_config.exogenous_scale
            factor *= quote_volatility(self.calibration, TICKS_PER_DAY) * scale / 0.0185
            if self.stock.tag == EXTREME_VOLATILITY_TAG:
                factor *= 1.25
            shock = factor * ((common_shock + sector_shock + local_shock) / scale + jump)
        shock_parts = {"全市場波動": factor * common_shock / scale,
                       "產業共同波動": factor * sector_shock / scale,
                       "個股擾動": factor * local_shock / scale, "突發跳空": factor * jump}
        if self.stock.tag == EXTREME_VOLATILITY_TAG:
            self.theme_impulse *= 0.72 ** (QUOTE_MINUTES / 15)
            if jump_roll < 0.55 / TICKS_PER_DAY:
                self.theme_impulse = math.copysign(0.015 + min(0.025, abs(jump_size) * 0.2), jump_size)
            theme_effect = self.theme_impulse * (QUOTE_MINUTES / 15) * (self.volatility / 1.5) * self.model_config.exogenous_scale
            if abs(theme_effect) >= 0.000001:
                shock += theme_effect
                shock_parts["題材資金脈衝"] = theme_effect
        scenario_move = 0.0
        if scenario_effect and self.stock.active and basket_value is None:
            scenario_name, scenario_move = scenario_effect
        if rate_pressure and self.stock.active and basket_value is None:
            rate_effect = rate_pressure * (QUOTE_MINUTES / 15) * RATE_SENSITIVITY.get(self.stock.sector, -0.65)
            shock += rate_effect
            shock_parts["模擬利率政策"] = rate_effect
        if basket_value is not None:
            self.fundamental = basket_value
        momentum = max(-1.0, min(1.0, (self.price / self.previous_price - 1) * 10))
        valuation = max(-1.5, min(1.5, (self.fundamental / self.price - 1) / 0.55))
        herd = max(-1.0, min(1.0, self.previous_imbalance))
        event = self.event
        decay = math.exp(-0.18 * (self.quote_tick - self.event_quote_tick) / TICKS_PER_DAY) if event else 0
        proposals = []
        base_scores, observed_news = [], []
        for a in self.agents:
            noise, verification_roll = self.rng.gauss(0, 0.34), self.rng.random()
            message = 0.0
            if event and a.ident in event.exposed:
                message = event.tone * event.power * decay * a.trust
                if event.truth == "虛假" and verification_roll < a.verify:
                    message *= 0.05
                if event.anchor == "偏多":
                    message += 0.16 * event.power * decay
                elif event.anchor == "偏空":
                    message -= 0.16 * event.power * decay
            w = STYLES[a.style]
            features = {
                "消息": w["news"] * message, "價格動能": w["trend"] * momentum,
                "估值": w["value"] * valuation, "羊群": w["herd"] * herd,
                "市場情緒": self.sentiment * 0.45, "噪音": noise,
            }
            score = sum(features.values()) - a.fear * max(0, -momentum) * (0.4 if a.style == "恐慌型" else 0.05)
            a.estimate = score
            threshold = 0.20 if a.style in ("恐慌型", "從眾型", "消息型") else 0.35
            quantity = min(5, max(1, int(abs(score) * 3))) if abs(score) > threshold else 0
            qty = quantity if score > threshold else -quantity if score < -threshold else 0
            qty = min(qty, max(0, int(a.cash // upper))) if qty > 0 else max(qty, -max(0, a.shares))
            proposals.append((a, qty, max(features, key=lambda k: abs(features[k]))))
            base_scores.append(score)
            observed_news.append(message)
        agent_limits = [None] * self.n
        if self.model_config.mode != "legacy":
            past_prices = [q["股價"] for q in self.quotes[-20:]] or [self.price]
            observation = Observation(self.price, self.stock.initial_price, self.fundamental,
                                      self.price / self.previous_price - 1,
                                      sum(past_prices[-5:]) / len(past_prices[-5:]),
                                      sum(past_prices) / len(past_prices), self.return_variance,
                                      lower, upper, self.quote_tick)
            decision = self.research_engine.decide(self.agents, observation, base_scores, observed_news)
            proposals = [(a, int(q), reason) for a, q, reason in zip(self.agents, decision.quantities, decision.reasons)]
            agent_limits = [None if math.isnan(value) else float(value) for value in decision.limits]
            for a, score in zip(self.agents, decision.scores):
                a.estimate = float(score)
        followed = 0
        activity = 1.0
        if self.realistic:
            activity, attention = self.participation(event)
            adjusted = []
            for i, (agent, qty, reason) in enumerate(proposals):
                roll = self.activity_rng.random()
                if roll >= activity:
                    qty, reason = 0, "等待流動性"
                elif (attention and agent.ident in event.exposed and roll < attention * agent.trust
                      and agent.style in ("從眾型", "消息型", "動能型")
                      and (self.model_config.mode == "legacy" or not self.research_engine.zi_mask[i])):
                    desired = 2 + round(10 * attention)
                    qty = min(desired, max(0, int(agent.cash // upper))) if event.tone > 0 else -min(desired, agent.shares)
                    if qty:
                        reason = "跟隨分析師"
                        agent_limits[i] = None
                        followed += abs(qty)
                adjusted.append((agent, qty, reason))
            proposals = adjusted
            if not activity:
                shock = 0.0
                shock_parts.clear()
        # ZI random limits are passive: unfilled orders do not form order flow.
        execution = None
        if self.realistic:
            prior_volume = self.quotes[-1]["成交量"] if self.quotes else self.depth
            execution = (max(1, round(self.depth * activity)), prior_volume, lower, TICKS_PER_DAY)
        institutional_plans = self.institutions.propose(
            self.institution_config, self.price, upper, self.fundamental,
            self.price / self.previous_price - 1,
            self.price / (sum(q["股價"] for q in self.quotes[-20:]) / len(self.quotes[-20:])) - 1 if self.quotes else 0,
            sum(observed_news) / self.n, common_shock, self.quote_tick, self.stock.taiwan_weight_pct, execution=execution)
        if not activity:
            institutional_plans = []
        institutional_net = sum(q for _, q, _ in institutional_plans)
        retail_net = sum(q for (_, q, _), limit in zip(proposals, agent_limits) if limit is None) + player_net
        net = retail_net + institutional_net
        old_price = self.price
        price_forming_net = net
        # Order flow is a drift per session; shocks scale with sqrt(time).
        impact_factor = 0.085 / (self.n * self.stock.liquidity * TICKS_PER_DAY)
        if self.realistic:
            impact_factor *= self.n * self.stock.liquidity / self.depth
        retail_impact = retail_net * impact_factor
        if self.stock.taiwan_weight_pct >= LARGE_CAP_WEIGHT_PCT:
            depth = min(self.stock.taiwan_weight_pct, 10)
            retail_cap = 0.002 / depth
            retail_impact = retail_cap * math.tanh(retail_impact / ((1 + 2 * depth) * retail_cap))
        order_parts = {"主動買賣需求": retail_impact}
        if self.institution_config.enabled:
            order_parts = {"一般代理人與玩家": retail_impact}
            for group in GROUPS:
                group_net = sum(q for account, q, _ in institutional_plans if account.group == group)
                multiplier = FOREIGN_PRICE_IMPACT_MULTIPLIER if group == "外資" else 1
                order_parts[f"{group}買賣力道"] = group_net * impact_factor * multiplier
        if self.realistic and self.calibration:
            # Small simulated populations can put an outsized fraction of the
            # entire market on one side. Scale flow to the calibrated price regime.
            flow_budget = 0.25 * quote_volatility(self.calibration, TICKS_PER_DAY)
            gross_impact = sum(abs(value) for value in order_parts.values())
            if gross_impact:
                flow_scale = flow_budget * math.tanh(gross_impact / flow_budget) / gross_impact
                order_parts = {name: value * flow_scale for name, value in order_parts.items()}
        impact = sum(order_parts.values())
        if self.stock.code == "2330" and impact + shock:
            # The opening print can reflect overnight repricing. Intraday
            # liquidity absorbs ordinary shocks from a single intraday quote.
            limit = math.log1p(0.02 if self.phase == 0 else 0.01)
            move = impact + shock
            damping = limit * math.tanh(move / limit) / move
            impact *= damping
            shock *= damping
            order_parts = {name: value * damping for name, value in order_parts.items()}
            shock_parts = {name: value * damping for name, value in shock_parts.items()}
        if scenario_move:
            # A market event reprices the stock independently of ordinary quote noise.
            shock += scenario_move
            shock_parts[f"重大事件・{scenario_name}"] = scenario_move
        if basket_value is None:
            change = max(-1, min(1, impact + shock))
            raw_target = old_price * math.exp(change)
            self.price = min(upper, max(lower, round(raw_target, 2)))
        else:
            basket_return = basket_value / (basket_previous_close or basket_value) - 1
            reference = basket_value
            premium = ETF_TRACKING_BAND * math.tanh((impact + shock) / ETF_TRACKING_BAND)
            raw_target = reference * (1 + premium)
            # ETF orders alone cannot move the print onto either daily limit.
            floor = round(lower + 0.01, 2) if reference > lower + 1e-8 and lower < upper else lower
            ceiling = round(upper - 0.01, 2) if reference < upper - 1e-8 and lower < upper else upper
            self.price = (min(ceiling, max(floor, round(raw_target, 2))) if floor <= ceiling
                          else min(upper, max(lower, round(reference, 2))))
            order_parts = {"ETF 成分籃子": math.log(reference / old_price),
                           "ETF 買賣與雜訊偏差": math.log1p(premium)}
            shock_parts = {}
        self.matched_limits = []
        self.player_fills = []
        filled_player_net = 0
        if self.realistic:
            book_lower, book_upper = (floor, ceiling) if basket_value is not None and floor <= ceiling else (lower, upper)
            depth = max(1, round(self.depth * (0.45 + activity)))
            self.book = (build_book(self.price, book_lower, book_upper, bool(self.stock.basket),
                                    depth, self.phase, TICKS_PER_DAY) if activity else {"買入": [], "賣出": []})
            self.player_fills = match_player(self.book, market_orders, limit_orders)
            filled_player_net = sum(quantity if order.side == "買入" else -quantity
                                    for order, _, quantity in self.player_fills)
            net += filled_player_net
            price_forming_net += filled_player_net
            if filled_player_net:
                # A fill consumes liquidity. Convert only executed flow into
                # bounded price pressure, leaving the matched prices intact.
                cap = min(0.004, 0.65 * quote_volatility(self.calibration, TICKS_PER_DAY)) if self.calibration else 0.002
                if self.stock.taiwan_weight_pct >= LARGE_CAP_WEIGHT_PCT:
                    cap = min(cap, 0.006 / min(self.stock.taiwan_weight_pct, 10))
                pressure = cap * math.tanh(filled_player_net / (2 * depth))
                if basket_value is None:
                    move = math.log(raw_target / old_price)
                    updated_move = move + pressure
                    if self.stock.code == "2330":
                        max_move = math.log1p(0.02 if self.phase == 0 else 0.01)
                        ordinary_move = updated_move - scenario_move
                        updated_move = scenario_move + max(-max_move, min(max_move, ordinary_move))
                    contribution = updated_move - move
                    raw_target = old_price * math.exp(updated_move)
                    self.price = min(upper, max(lower, round(raw_target, 2)))
                else:
                    updated_premium = ETF_TRACKING_BAND * math.tanh((impact + shock + pressure) / ETF_TRACKING_BAND)
                    updated_target = reference * (1 + updated_premium)
                    contribution = math.log(updated_target / raw_target)
                    raw_target = updated_target
                    if floor <= ceiling:
                        self.price = min(ceiling, max(floor, round(raw_target, 2)))
                order_parts["玩家成交買賣力道"] = contribution
                impact += contribution
        self.last_explanation = explain_move(old_price, self.price, {**order_parts, **shock_parts},
                                             lower, upper, raw_target, price_forming_net,
                                             event.title if event else "", event.tone if event else 0,
                                             decay if event and event.power > 0 else 0)
        explanation_fields = {key: self.last_explanation[key] for key in ("本筆漲跌 (%)", "漲跌說明", "新聞脈絡", "價格限制")}
        explanation_fields.update({f"{key}貢獻 (百分點)": value for key, value in self.last_explanation["分解"].items()})
        explanation_fields.update({"前價": old_price, "主動淨需求": price_forming_net,
                                   "作用新聞": event.title if event and event.power > 0 and decay >= 0.01 else ""})
        if self.realistic:
            explanation_fields["玩家本筆淨成交股數"] = filled_player_net
            if followed:
                explanation_fields["分析師跟單股數"] = followed
            if not activity:
                explanation_fields["流動性狀態"] = "冷清無成交"
        if basket_value is not None:
            explanation_fields.update({"ETF 籃子當日漲跌 (%)": basket_return * 100,
                                       "ETF 追蹤參考價": reference,
                                       "ETF 折溢價 (%)": (self.price / reference - 1) * 100})
        executed = []
        for (a, qty, reason), limit in zip(proposals, agent_limits):
            if limit is not None:
                fills = (qty > 0 and self.price <= limit) or (qty < 0 and self.price >= limit)
                qty = qty if fills else 0
                net += qty
                if not fills:
                    reason = "ZI 限價未觸及"
            executed.append((a, qty, reason))
        proposals = executed
        buys = sum(q > 0 for _, q, _ in proposals)
        sells = sum(q < 0 for _, q, _ in proposals)
        if not self.realistic:
            self.matched_limits = [o for o in limit_orders if
                                   (o.side == "買入" and self.price <= o.limit_price) or
                                   (o.side == "賣出" and self.price >= o.limit_price)]
            net += sum(o.remaining if o.side == "買入" else -o.remaining for o in self.matched_limits)
            self.player_fills = [(o, self.price, o.remaining)
                                 for o in sorted((*market_orders, *self.matched_limits), key=lambda order: order.ident)]
        other_fills = [(account.sleeve, qty, None) for account, qty, _ in institutional_plans]
        for order, execution_price, quantity in self.player_fills:
            other_fills.append(("玩家", quantity if order.side == "買入" else -quantity, order.ident, execution_price))
        if not market_orders and player_volume:
            bought = (player_volume + player_net) // 2
            sold = player_volume - bought
            other_fills.extend(("模擬委託", qty, None) for qty in (bought, -sold) if qty)
        volume = sum(abs(qty) for _, qty, _ in proposals) + sum(abs(fill[1]) for fill in other_fills)
        turnover = round(self.price * (volume - sum(q for _, _, q in self.player_fills)) +
                         sum(price * q for _, price, q in self.player_fills), 2)
        self.previous_price = old_price
        self.previous_imbalance = net / max(1, volume)
        self.quote_tick += 1
        self.phase += 1
        self.institutions.settle(institutional_plans, self.price)
        institutional_fields = {f"{group}本筆買賣超": sum(q for account, q, _ in institutional_plans if account.group == group) for group in GROUPS}
        institutional_fields["三大法人本筆買賣超"] = institutional_net
        realized_return = self.price / old_price - 1
        self.return_variance = 0.95 * self.return_variance + 0.05 * realized_return ** 2
        if self.model_config.mode != "legacy":
            self.research_engine.observe(realized_return, self.quote_tick)
        for index, (a, qty, reason) in enumerate(proposals):
            if qty > 0:
                a.cost_basis = (a.cost_basis * a.shares + qty * self.price) / (a.shares + qty)
            a.cash = round(a.cash - qty * self.price, 2)
            a.shares += qty
            if a.shares == 0:
                a.cost_basis = self.price
            a.peak_wealth = max(a.peak_wealth, a.wealth(self.price))
            a.forecast_error = (realized_return - a.belief) ** 2 if self.model_config.mode != "legacy" and a.model_family != "ZI-C 簡化" else None
            a.last_qty, a.reason = qty, reason
            a.action = "買入" if qty > 0 else "賣出" if qty < 0 else "觀望"
            a.fear = min(1, max(0, a.fear + (0.08 if self.price < old_price else -0.025) / scale))
            self.daily_agent_volume[index] += abs(qty)
        record = self.history[-1]
        if self.phase == 1 or not self.quotes:
            record["開盤"] = record["最高"] = record["最低"] = self.price
        record.update({
            "最高": max(record["最高"], self.price), "最低": min(record["最低"], self.price),
            "收盤": self.price, "股價": self.price,
            "漲跌幅 (%)": round((self.price / self.previous_close - 1) * 100, 6),
            "成交量": record["成交量"] + volume, "淨需求": record["淨需求"] + net,
            "買入人數": buys, "賣出人數": sells, "觀望人數": self.n - buys - sells,
            "市場衝擊 (%)": record["市場衝擊 (%)"] + shock * 100,
            "模型": self.model_config.mode,
            "訂單衝擊 (%)": record.get("訂單衝擊 (%)", 0) + impact * 100,
            "預測MSE": self.research_engine.forecast_mse if self.model_config.mode != "legacy" else None,
            **explanation_fields,
            **{f"{group}當日買賣超": sum(row["買賣超股數"] for row in self.institutions.current.values() if row["法人"] == group) for group in GROUPS},
        })
        self.quotes.append({"交易日": self.trading_date.isoformat(), "時間": self.clock, "報價序號": self.quote_tick,
                            "股價": self.price, "成交量": volume, "成交金額": turnover, "漲跌幅 (%)": record["漲跌幅 (%)"],
                            "模型": self.model_config.mode,
                            "預測MSE": record["預測MSE"], "訂單衝擊 (%)": impact * 100,
                            "外部衝擊 (%)": shock * 100, **explanation_fields, **institutional_fields})
        self.fill_batches[self.quote_tick] = (bytes(qty + 128 for _, qty, _ in proposals), tuple(other_fills))
        if self.phase == TICKS_PER_DAY:
            self.institutions.close_day()
            record["狀態"] = "收盤"
            self.tick += 1
            day_logs = [
                {
                    "回合": self.tick, "交易日": self.trading_date.isoformat(), "代理人": a.ident,
                    "類型": a.style, "最後動作": a.action, "當日交易股數": self.daily_agent_volume[index],
                    "主要原因": a.reason, "持有股數": a.shares, "現金": round(a.cash, 2),
                    "資產": round(a.wealth(self.price), 2),
                    "模型": a.model_family, "人物設定": a.personality, "信念": round(a.belief, 6),
                    "成本錨點": round(a.cost_basis, 4), "預測誤差": a.forecast_error,
                }
                for index, a in enumerate(self.agents)
            ]
            self.log_batches.append(self.pack_logs(day_logs))
            self.phase = 0
        return record

    @staticmethod
    def pack_logs(rows):
        return zlib.compress(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), level=1)

    def iter_logs(self):
        for batch in self.log_batches:
            yield from json.loads(zlib.decompress(batch))

    def trade_records(self, day):
        """Expand the stored fills for one session into a two-sided ledger."""
        for tick, quote in enumerate(self.quotes, 1):
            tick = quote.get("報價序號", tick)
            if quote["交易日"] != day or tick not in self.fill_batches:
                continue
            packed, other = self.fill_batches[tick]
            fills = [(f"AI #{agent.ident}", qty - 128, None)
                     for agent, qty in zip(self.agents, packed) if qty != 128]
            for seq, (party, qty, order_id, *execution_price) in enumerate((*fills, *other), 1):
                price = execution_price[0] if execution_price else quote["股價"]
                yield {"成交編號": f"{self.stock.code or self.stock.symbol}-{quote.get('原成交序號', tick)}-{seq}",
                       "交易日": day, "時間": quote["時間"], "代號": self.stock.code,
                       "買方": party if qty > 0 else "做市商",
                       "賣方": "做市商" if qty > 0 else party, "成交股數": abs(qty),
                       "成交價": price, "成交金額": round(abs(qty) * price, 2),
                       "玩家委託編號": order_id}

    def step(self, common_shock=0.0, sector_shock=0.0):
        target = self.tick + 1
        while self.tick < target:
            self.quote(common_shock, sector_shock)
        return self.history[-1]

    def advance(self, steps):
        for _ in range(steps):
            self.step()


class WeightedIndex:
    """Fixed synthetic share basket, with drifting current constituent weights."""

    def __init__(self, markets, weights=None, base_points=10000.0):
        values = {code: INDEX_WEIGHTS.get(code, 0.0) for code in markets} if weights is None else weights
        if set(values) != set(markets):
            raise ValueError("權重必須包含所有股票。")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("權重必須為非負的有限數字。")
        if not math.isclose(sum(values.values()), 100, abs_tol=0.001):
            raise ValueError("基期權重合計必須為 100%。")
        total = sum(values.values())
        self.weights = {code: value / total for code, value in values.items()}
        self.bases = {code: market.stock.initial_price for code, market in markets.items()}
        self.base_points = self.value = self.previous_close = base_points
        first = next(iter(markets.values())).history[0]
        self.history = [{"交易日": first["交易日"], "狀態": "起點", "昨收": base_points,
                         "開盤": base_points, "最高": base_points, "最低": base_points,
                         "收盤": base_points, "股價": base_points, "成交量": 0, "漲跌幅 (%)": 0.0}]
        self.quotes = []

    def calculate(self, prices):
        return self.base_points * sum(self.weights[code] * price / self.bases[code] for code, price in prices.items())

    def record(self, day, clock, prices, volume, *, store_quote=True):
        new_day = day != self.history[-1]["交易日"]
        if new_day:
            self.previous_close = self.value
        self.value = self.calculate(prices)
        if new_day:
            self.history.append({"交易日": day, "狀態": "盤中", "昨收": self.previous_close,
                                 "開盤": self.value, "最高": self.value, "最低": self.value,
                                 "收盤": self.value, "股價": self.value, "成交量": 0})
        row = self.history[-1]
        row.update({"最高": max(row["最高"], self.value), "最低": min(row["最低"], self.value),
                    "收盤": self.value, "股價": self.value, "成交量": row["成交量"] + volume,
                    "漲跌幅 (%)": (self.value / self.previous_close - 1) * 100,
                    "狀態": "收盤" if clock == "13:30" else "盤中"})
        if store_quote:
            self.quotes.append({"交易日": day, "時間": clock, "股價": self.value, "成交量": volume})

    def composition(self, markets):
        rows = []
        for code, market in markets.items():
            value = self.base_points * self.weights[code] * market.price / self.bases[code]
            contribution = self.base_points * self.weights[code] * (market.price - market.history[-1]["昨收"]) / self.bases[code]
            rows.append({"代碼": code, "股票": market.stock.name, "產業": market.stock.sector,
                         "台股代號": market.stock.code, "台股參考占比 (%)": market.stock.taiwan_weight_pct,
                         "基期權重 (%)": self.weights[code] * 100,
                         "指數占比 (%)": value / self.value * 100, "貢獻點數": contribution})
        return rows


class Exchange:
    """Weekday sessions, live prints and a shared player portfolio."""
    version = MODEL_VERSION

    def __init__(self, n=100, seed=42, volatility=1.5, start_date=date(2026, 11, 2),
                 model_config=None, institution_config=None, index_base=10000.0):
        self.seed, self.tick, self.volatility = seed, 0, volatility
        self.quote_minutes = QUOTE_MINUTES
        self.rng = random.Random(seed + 3000003)
        self.markets = {
            stock.symbol: Market(n, seed + (int(stock.symbol[2:]) - 1) * 1009, stock, volatility, start_date, model_config, institution_config)
            for stock in INSTRUMENTS
        }
        self.player = Player()
        self.index = WeightedIndex(self.markets, base_points=index_base)
        self.schema_version = MODEL_VERSION
        self.auto_running = False
        self.last_auto_time = None
        self.model_config = model_config or ModelConfig.preset("legacy")
        self.institution_config = next(iter(self.markets.values())).institution_config
        self.auto_news_config = AutoNewsConfig()
        self.random_news_rng = random.Random(seed + 7000003)
        self.news_feed = []
        self.initialize_scenarios()
        self.initialize_rate_policy()

    def enable_realistic_execution(self):
        for market in self.markets.values():
            market.realistic = True
        for order in self.player.pending:
            if order.side == "買入" and not getattr(order, "includes_fee", False):
                reserve = round(order.reserved_cash + fee(order.reserved_cash), 2)
                if reserve - order.reserved_cash <= self.player.available_cash + 1e-8:
                    order.reserved_cash = reserve
                    order.includes_fee = True
                else:
                    order.status = "升級取消（費用預留不足）"

    def seed_daily_history(self, history):
        """Start a fresh exchange at the close of imported daily records."""
        days = sorted({row["交易日"] for rows in history.values() for row in rows})
        if not days or self.tick or self.phase or self.player.orders:
            raise ValueError("請用有效行情建立接續場次。")
        if any(date.fromisoformat(day).weekday() >= 5 for day in days):
            raise ValueError("行情中有週末日期，請確認使用 AFS 匯出的交易日資料。")
        self.imported_days = tuple(days)
        self.tick = len(days)
        day_number = {day: number for number, day in enumerate(days, 1)}
        baseline_day = date.fromisoformat(days[0]) - timedelta(days=1)
        last_day = date.fromisoformat(days[-1])
        for code, market in self.markets.items():
            market.trading_date = baseline_day
            market.history = [market._record("起點")]
            prior = market.price
            for bar in history.get(code, ()):
                day = bar["交易日"]
                close = bar["收盤"]
                previous_close = bar.get("昨收", prior)
                if not math.isfinite(previous_close) or previous_close <= 0:
                    previous_close = prior
                lower, upper = market.trading_limits(previous_close)
                market.history.append({
                    "回合": day_number[day], "交易日": day, "狀態": "收盤",
                    "開盤": bar["開盤"], "最高": bar["最高"], "最低": bar["最低"],
                    "收盤": close, "股價": close, "昨收": previous_close, "成交量": int(bar["成交量"]),
                    "漲跌幅 (%)": (close / previous_close - 1) * 100,
                    "跌停價": lower, "漲停價": upper,
                })
                prior = close
            market.price = market.previous_price = market.fundamental = prior
            market.previous_close = market.history[-1]["昨收"]
            market.trading_date = last_day
            market.tick = self.tick
            market.quote_tick = self.tick * TICKS_PER_DAY
            for agent in market.agents:
                agent.cost_basis = prior
                agent.peak_wealth = agent.wealth(prior)
        for code in history:
            market = self.markets[code]
            stock = market.stock
            if stock.basket and stock.tag != ACTIVE_ETF_TAG:
                # Rebase the fixed basket to the imported close. CSVs lack the prior NAV.
                units = [(symbol, stock.initial_price * weight / self.markets[symbol].stock.initial_price)
                         for symbol, weight in stock.basket]
                nav = sum(shares * self.markets[symbol].price for symbol, shares in units)
                market.etf_allocation = ETFAllocation(last_day, tuple(
                    (symbol, shares * market.price / nav) for symbol, shares in units
                ))
        self.reweight_index({code: weight * 100 for code, weight in self.index.weights.items()})

    def initialize_scenarios(self):
        self.auto_scenarios = False
        self.scenario = None
        self.scenario_rng = random.Random(self.seed + 10000019)

    def scenario_at(self, quote_tick):
        if self.scenario is None:
            return None
        name, started = self.scenario
        if 0 <= quote_tick - started < SCENARIO_DURATION_DAYS * TICKS_PER_DAY:
            return name
        return None

    def start_scenario(self, name):
        scenario = MARKET_SCENARIOS[name]
        tick = max(market.quote_tick for market in self.markets.values())
        if self.scenario_at(tick):
            raise ValueError("已有重大事件正在影響行情，請等影響期結束後再觸發。")
        self.scenario = (name, tick)
        codes = [code for code, market in self.markets.items() if market.stock.active]
        source = "模擬極端事件" if scenario.opening_aftershocks else "模擬重大事件"
        event = {"發布時間": self.next_timestamp, "來源": source, "影響範圍": "全市場",
                 "方向": "偏多" if scenario.opening_move > 0 else "偏空",
                 "標的數": len(codes), "標的代碼": ",".join(codes),
                 "消息": f"【{source}】{name}｜{scenario.headline}"}
        self.news_feed.append(event)
        return event

    def scenario_effect(self, market):
        name = self.scenario_at(market.quote_tick)
        if name is None:
            return None
        elapsed = market.quote_tick - self.scenario[1]
        scenario = MARKET_SCENARIOS[name]
        decay = 0.5 ** (elapsed / TICKS_PER_DAY)
        effect = scenario.daily_pressure / TICKS_PER_DAY * decay
        if elapsed == 0:
            effect += scenario.opening_move
        elif market.phase == 0:
            # Also handle events triggered during a session rather than at its open.
            opening = (elapsed - 1) // TICKS_PER_DAY
            if opening < len(scenario.opening_aftershocks):
                effect += scenario.opening_aftershocks[opening]
        return name, effect * scenario.sector_exposure.get(market.stock.sector, 1.0)

    def initialize_rate_policy(self):
        self.policy_rate_bps = 200
        self.rate_meeting_interval = 0
        self.rate_pressure_initial = 0.0
        self.rate_pressure_tick = 0
        self.rate_rng = random.Random(self.seed + 9000003)

    def policy_pressure(self, quote_tick):
        elapsed = max(0, quote_tick - self.rate_pressure_tick)
        if elapsed >= 6 * TICKS_PER_DAY:
            return 0.0
        return self.rate_pressure_initial * 0.5 ** (elapsed / TICKS_PER_DAY)

    def apply_rate_decision(self, basis_points, *, automatic=False):
        if type(basis_points) is not int or basis_points not in (-50, -25, 0, 25, 50):
            raise ValueError("利率決議須為升降 25、50 個基點或維持利率。")
        before = self.policy_rate_bps
        after = max(0, min(1000, before + basis_points))
        change = after - before
        if change:
            tick = max(market.quote_tick for market in self.markets.values())
            combined = self.policy_pressure(tick) + change * 0.000018
            self.rate_pressure_initial = max(-0.0018, min(0.0018, combined))
            self.rate_pressure_tick = tick
        self.policy_rate_bps = after
        action = f"升息 {change} 基點" if change > 0 else f"降息 {-change} 基點" if change < 0 else "利率維持不變"
        codes = [code for code, market in self.markets.items() if market.stock.active]
        event = {"發布時間": self.next_timestamp, "來源": "隨機利率會議" if automatic else "手動利率決議",
                 "影響範圍": "全市場", "方向": "依產業而異" if change else "中性",
                 "標的數": len(codes), "標的代碼": ",".join(codes), "利率變動 (bp)": change,
                 "模擬利率 (%)": after / 100,
                 "消息": f"【模擬利率】央行{action}，政策利率 {before / 100:.2f}% → {after / 100:.2f}%"}
        self.news_feed.append(event)
        return event

    def configure_news(self, config):
        if not isinstance(config, AutoNewsConfig):
            raise ValueError("Invalid news configuration")
        self.auto_news_config = config

    @property
    def next_timestamp(self):
        market = next(iter(self.markets.values()))
        if market.phase == 0:
            day = market.start_date if market.tick == 0 else next_weekday(market.trading_date)
            return f"{day.isoformat()} 09:{QUOTE_MINUTES:02d}"
        minutes = 9 * 60 + (market.phase + 1) * QUOTE_MINUTES
        return f"{market.trading_date.isoformat()} {minutes // 60:02d}:{minutes % 60:02d}"

    def publish_analyst_call(self, symbol, tone):
        stock = self.markets[symbol].stock
        self.announce(analyst_headline(stock, tone), tone, 90, 85, "未知", "中立", symbol,
                      source="模擬分析師", published_at=self.next_timestamp)
        return self.news_feed[-1]

    def publish_price_forecast(self, symbol, target_price):
        market = self.markets.get(symbol)
        if market is None or not market.stock.active:
            raise ValueError("請選擇一檔可觀察的股票或 ETF。")
        if (isinstance(target_price, bool) or not isinstance(target_price, (int, float))
                or not 0.01 <= target_price <= 1_000_000_000):
            raise ValueError("目標價須介於 0.01 元與 10 億元之間。")

        current = market.price
        target_price = round(target_price, 2)
        upside = target_price / current - 1
        if not math.isfinite(upside):
            raise ValueError("目標價超過可計算的範圍。")
        tone = (upside > 0) - (upside < 0)
        # A large forecast increases attention, up to the analyst model's cap.
        power = min(90, 30 + round(min(abs(upside), 0.25) * 240)) if tone else 0
        headline = (f"【模擬目標價】{market.stock.code} {market.stock.name} 目標 {target_price:,.2f} 元"
                    f"｜發布時參考價 {current:,.2f} 元（{upside:+.2%}）")
        self.announce(headline, tone, power, 85, "未知", "中立", symbol,
                      source="模擬分析師", published_at=self.next_timestamp)
        event = self.news_feed[-1]
        event.update({"目標價（元）": target_price, "發布時參考價（元）": current})
        return event

    def generate_random_news(self, force=False, upcoming=False):
        news = draw_news(self.random_news_rng, self.auto_news_config,
                         [m.stock for m in self.markets.values() if m.stock.active], force)
        if news is None:
            return None
        targets = [code for code, m in self.markets.items() if m.stock.active and
                   (news["scope"] == "全市場" or
                    news["scope"] == "產業" and m.stock.sector == news["stock"].sector or
                    news["scope"] == "個股" and code == news["stock"].symbol)]
        if not force:
            targets = [code for code in targets if not (
                self.markets[code].event and getattr(self.markets[code].event, "source", "手動") in ("手動", "模擬分析師") and
                self.markets[code].quote_tick - self.markets[code].event_quote_tick < self.auto_news_config.manual_cooldown)]
        if not targets:
            return None
        scope = "全市場" if news["scope"] == "全市場" else news["stock"].sector if news["scope"] == "產業" else f"{news['stock'].code} {news['stock'].name}"
        self.announce(news["title"], news["tone"], news["power"], news["reach"], news["truth"], news["anchor"],
                      source="模擬分析師" if news.get("analyst") else "隨機（手動觸發）" if force else "隨機",
                      scope=scope, targets=targets,
                      published_at=self.next_timestamp if upcoming else self.timestamp)
        return self.news_feed[-1]

    def configure_model(self, config):
        for market in self.markets.values():
            market.configure_model(config)
        self.model_config = config
        self.institution_config = next(iter(self.markets.values())).institution_config

    def configure_institutions(self, config):
        for market in self.markets.values():
            market.configure_institutions(config)
        self.institution_config = next(iter(self.markets.values())).institution_config

    def institution_records(self, detail=False):
        return [{"模擬代碼": code, "台股代號": market.stock.code, "股票": market.stock.name, **row}
                for code, market in self.markets.items()
                for row in (market.institutions.history if detail else market.institutions.grouped_history())]

    @property
    def phase(self):
        return next(iter(self.markets.values())).phase

    @property
    def timestamp(self):
        market = next(iter(self.markets.values()))
        return f"{market.trading_date.isoformat()} {market.clock}"

    def place_order(self, symbol, side, quantity, order_type="市價", limit_price=None):
        if symbol not in self.markets:
            raise ValueError("請選擇有效股票。")
        if side not in ("買入", "賣出"):
            raise ValueError("請選擇買入或賣出。")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("股數必須是正整數。")
        if order_type not in ("市價", "限價"):
            raise ValueError("請選擇市價或限價。")
        market = self.markets[symbol]
        if not market.stock.active and side == "買入":
            raise ValueError("此標的已移出新清單，只供既有模擬部位賣出。")
        lower, upper = market.next_limits
        if order_type == "限價":
            if isinstance(limit_price, bool) or not isinstance(limit_price, (int, float)) or not math.isfinite(limit_price):
                raise ValueError("請輸入有效的買進或賣出價格。")
            if not lower <= limit_price <= upper:
                raise ValueError(f"委託價格必須介於跌停 {lower:,.2f} 與漲停 {upper:,.2f} 元。")
            cents = round(limit_price * 100)
            if market.realistic and cents % tick_cents(cents, bool(market.stock.basket)):
                raise ValueError("委託價格不符合此價位的跳動單位。")
            if Decimal(str(limit_price)) != Decimal(str(limit_price)).quantize(Decimal("0.01")):
                raise ValueError("委託價格最多兩位小數。")
        elif limit_price is not None:
            raise ValueError("市價單不需指定委託價格。")
        reserved = 0.0
        if side == "買入":
            reserved = round((limit_price if order_type == "限價" else upper) * quantity, 2)
            if market.realistic:
                reserved = round(reserved + fee(reserved), 2)
            if reserved > self.player.available_cash:
                raise ValueError(f"可用現金不足；此委託需預留 {reserved:,.2f} 元。")
        elif quantity > self.player.available_shares(symbol):
            raise ValueError("可賣股數不足，不能放空或重複委託同一批股票。")
        order = Order(len(self.player.orders) + 1, symbol, side, quantity,
                      reserved_cash=reserved, submitted_at=self.timestamp,
                      order_type=order_type, limit_price=limit_price,
                      valid_day=(market.start_date if market.tick == 0 else next_weekday(market.trading_date)).isoformat()
                      if market.phase == 0 else market.trading_date.isoformat())
        order.includes_fee = market.realistic
        self.player.orders.append(order)
        return order

    def cancel_order(self, ident):
        order = next((o for o in self.player.orders if o.ident == ident), None)
        if order is None or order not in self.player.pending:
            raise ValueError("此委託無法取消。")
        order.status = "已取消"

    def quote(self):
        # A previous exception may have left an exchange print half complete.
        # Finish lagging stocks before advancing those already settled; player
        # orders on completed stocks must not execute a second time.
        progress = [market.quote_tick for market in self.markets.values()]
        recovering = min(progress) != max(progress)
        first = next(iter(self.markets.values()))
        index_last = self.index.quotes[-1] if self.index.quotes else None
        imported_ticks = len(getattr(self, "imported_days", ())) * TICKS_PER_DAY
        rebuild_index = recovering or (first.quote_tick > imported_ticks and (
            index_last is None or f'{index_last["交易日"]} {index_last["時間"]}' != self.timestamp))
        target = max(progress) if recovering else max(progress) + 1
        if not recovering:
            self.generate_random_news(upcoming=True)
            if (self.rate_meeting_interval and first.phase == 0 and first.tick > 0
                    and first.tick % self.rate_meeting_interval == 0):
                decision = self.rate_rng.choices((-25, 0, 25), weights=(3, 4, 3))[0]
                self.apply_rate_decision(decision, automatic=True)
            if self.auto_scenarios and first.phase == 0 and not self.scenario_at(first.quote_tick):
                roll = self.scenario_rng.random()
                if roll < EXTREME_SCENARIO_CHANCE:
                    self.start_scenario(self.scenario_rng.choice(EXTREME_SCENARIOS))
                elif roll < EXTREME_SCENARIO_CHANCE + ORDINARY_SCENARIO_CHANCE:
                    self.start_scenario(self.scenario_rng.choice(ORDINARY_SCENARIOS))
        common = self.rng.gauss(0, 0.009)
        sectors = {sector: self.rng.gauss(0, 0.008) for sector in dict.fromkeys(m.stock.sector for m in self.markets.values())}
        for symbol, market in self.markets.items():
            while market.quote_tick < target:
                orders = [o for o in self.player.pending if o.symbol == symbol]
                market_orders = [o for o in orders if o.order_type == "市價"]
                limit_orders = [o for o in orders if o.order_type == "限價"]
                net = 0 if market.realistic else sum(o.remaining * (1 if o.side == "買入" else -1) for o in market_orders)
                market.volatility = self.volatility
                if market.stock.tag == ACTIVE_ETF_TAG and market.phase == 0 and market.tick:
                    self.rebalance_etf(market)
                market.quote(common, sectors[market.stock.sector], net, sum(o.remaining for o in market_orders), limit_orders,
                             market_orders=market_orders,
                             basket_value=self.basket_value(market.stock, market.quote_tick),
                             basket_previous_close=self.basket_value(
                                 market.stock, market.quote_tick - market.quote_tick % TICKS_PER_DAY - 1),
                             rate_pressure=self.policy_pressure(market.quote_tick),
                             scenario_effect=self.scenario_effect(market))
                for order, price, quantity in market.player_fills:
                    self.player.fill(order, price, f"{market.trading_date.isoformat()} {market.clock}",
                                     market.stock.name, quantity, stock=market.stock if market.realistic else None)
                if market.realistic:
                    for order in market_orders:
                        if order.remaining:
                            order.status = "部分成交餘量取消" if order.filled_quantity else "未成交已取消"
                if market.phase == 0:
                    for order in orders:
                        if order.status in ("待成交", "部分成交"):
                            order.status = "部分成交餘量到期" if order.filled_quantity else "已到期"
                market.player_fills = []
        self.tick = next(iter(self.markets.values())).tick
        if rebuild_index:
            self.reweight_index({code: weight * 100 for code, weight in self.index.weights.items()})
            return
        self.index.record(first.trading_date.isoformat(), first.clock,
                          {code: m.price for code, m in self.markets.items()},
                          sum(m.quotes[-1]["成交量"] for m in self.markets.values()))

    def component_price(self, symbol, quote_index):
        component = self.markets[symbol]
        if quote_index < 0:
            return component.stock.initial_price
        imported_days = getattr(self, "imported_days", ())
        day_index = quote_index // TICKS_PER_DAY
        if day_index < len(imported_days):
            end = bisect_right(component.history, imported_days[day_index], key=lambda row: row["交易日"])
            return component.history[end - 1]["收盤"] if end else component.stock.initial_price
        if len(component.quotes) != component.quote_tick:
            end = bisect_right(component.quotes, quote_index + 1, key=lambda quote: quote["報價序號"])
            return component.quotes[end - 1]["股價"] if end else component.stock.initial_price
        return component.quotes[quote_index]["股價"] if quote_index < len(component.quotes) else component.price

    def rebalance_etf(self, market):
        day = next_weekday(market.trading_date)
        if market.etf_allocation and market.etf_allocation.day == day:
            return
        previous = market.quote_tick - 1
        sessions = min(market.tick, 5)
        risk_penalty = 1.0 if market.stock.code == "00984A" else 0.25
        tilt = 0.25 if market.stock.code == "00985A" else 0.60
        targets = []
        for symbol, base_weight in market.stock.basket:
            closes = [self.component_price(symbol, previous - offset * TICKS_PER_DAY)
                      for offset in range(sessions, -1, -1)]
            returns = [math.log(right / left) for left, right in zip(closes, closes[1:])]
            score = sum(returns) - risk_penalty * sum(abs(change) for change in returns) / sessions
            targets.append((symbol, base_weight * (1 + tilt * math.tanh(score / 0.05)), closes[-1]))
        total = sum(target for _, target, _ in targets)
        # Buy the new mix at the previous close using the existing fund NAV.
        market.etf_allocation = ETFAllocation(day, tuple(
            (symbol, market.fundamental * target / (total * close))
            for symbol, target, close in targets
        ))

    def etf_holdings_value(self, stock, quote_index):
        allocation = self.markets[stock.symbol].etf_allocation
        units = allocation.units if allocation else (
            (symbol, stock.initial_price * weight / self.markets[symbol].stock.initial_price)
            for symbol, weight in stock.basket
        )
        return [(symbol, shares * self.component_price(symbol, quote_index)) for symbol, shares in units]

    def basket_value(self, stock, quote_index):
        if not stock.basket:
            return None
        return sum(value for _, value in self.etf_holdings_value(stock, quote_index))

    def reweight_index(self, weights):
        index = WeightedIndex(self.markets, weights, base_points=getattr(getattr(self, "index", None), "base_points", 10000.0))
        # An interrupted settlement can leave a missing historical print even
        # after all simulation clocks catch up. Align recorded prints by time.
        timeline = defaultdict(dict)
        for code, market in self.markets.items():
            for quote in market.quotes:
                timeline[quote["交易日"], quote["時間"]][code] = quote
        prices = dict(index.bases)
        imported_days = getattr(self, "imported_days", ())
        if imported_days:
            bars_by_day = defaultdict(dict)
            for code, market in self.markets.items():
                for row in market.history[1:]:
                    if row["交易日"] > imported_days[-1]:
                        break
                    bars_by_day[row["交易日"]][code] = row
            for day in imported_days:
                bars = bars_by_day[day]
                prices.update({code: row["開盤"] for code, row in bars.items()})
                index.record(day, "09:05", prices, 0, store_quote=False)
                prices.update({code: row["收盤"] for code, row in bars.items()})
                index.record(day, "13:30", prices, sum(row["成交量"] for row in bars.values()), store_quote=False)
        for (day, clock), prints in sorted(timeline.items()):
            prices.update({code: quote["股價"] for code, quote in prints.items()})
            index.record(day, clock, prices, sum(quote["成交量"] for quote in prints.values()))
        self.index = index

    def auto_quote(self, now, interval, enabled):
        """Advance once when due; UI reruns never accelerate or catch up time."""
        if not enabled or not self.auto_running or self.last_auto_time is None:
            self.auto_running = enabled
            self.last_auto_time = now
            return False
        if now - self.last_auto_time < interval:
            return False
        self.quote()
        self.last_auto_time = now
        return True

    def advance(self, steps):
        target = self.tick + steps
        while self.tick < target:
            self.quote()

    def announce(self, title, tone, power, reach, truth, anchor, target=None, *, source="手動", scope=None, targets=None, published_at=None):
        codes = list(self.markets) if targets is None and target is None else [target] if targets is None else targets
        timestamp = published_at or self.timestamp
        scope = scope or ("全市場" if target is None else f"{self.markets[target].stock.code} {self.markets[target].stock.name}")
        for code in codes:
            self.markets[code].announce(title, tone, power, reach, truth, anchor,
                                        source=source, published_at=timestamp, scope=scope)
        self.news_feed.append({"發布時間": timestamp, "來源": source, "影響範圍": scope,
                               "方向": "利多" if tone > 0 else "利空" if tone < 0 else "中性",
                               "強度": power, "觸及率 (%)": reach, "標的數": len(codes),
                               "標的代碼": ",".join(codes), "消息": title.strip()})

    def records(self, kind):
        return list(self.iter_records(kind))

    def iter_records(self, kind):
        if kind not in ("history", "logs", "events", "quotes"):
            raise ValueError("Unknown record kind")
        for symbol, market in self.markets.items():
            rows = market.iter_logs() if kind == "logs" else getattr(market, kind)
            for row in rows:
                yield {"股票代碼": symbol, "台股代號": market.stock.code, "股票名稱": market.stock.name, **row}


def upgrade_exchange(world):
    """Expand a live world while preserving existing cash, holdings and prints."""
    world.__class__ = Exchange
    world.player.__class__ = Player
    if not hasattr(world.player, "intraday_inventory"):
        player = world.player
        player.intraday_inventory = {}
        player.inventory_day = player.trades[-1]["成交時間"][:10] if player.trades else None
        latest_trades = []
        for trade in reversed(player.trades):
            if not trade["成交時間"].startswith(player.inventory_day):
                break
            latest_trades.append(trade)
        for trade in reversed(latest_trades):
            symbol = trade["股票代碼"]
            previous = player.intraday_inventory.get(symbol, 0)
            change = trade["股數"] if trade["買賣"] == "買入" else -trade["股數"]
            player.intraday_inventory[symbol] = max(0, previous + change)
    if not hasattr(world.player, "deposited_cash"):
        world.player.deposited_cash = 0.0
    first = next(iter(world.markets.values()))
    config = getattr(world, "model_config", ModelConfig.preset("hybrid"))
    institutional_config = getattr(world, "institution_config", InstitutionConfig(enabled=True))
    if not isinstance(institutional_config, InstitutionConfig):
        institutional_config = InstitutionConfig(enabled=institutional_config.enabled, activity=institutional_config.activity)
    if not isinstance(config, ModelConfig):
        config = ModelConfig(**config.to_dict())
    old_weights = dict(world.index.weights) if hasattr(world, "index") else {}
    old_sectors = list(dict.fromkeys(m.stock.sector for m in world.markets.values()))
    known = {stock.symbol: stock for stock in INSTRUMENTS}
    for market in world.markets.values():
        market.__class__ = Market
        if not hasattr(market, "etf_allocation"):
            market.etf_allocation = None
        if not hasattr(market, "theme_impulse"):
            market.theme_impulse = 0.0
        if not hasattr(market, "realistic"):
            market.realistic = False
            market.book = None
        if not hasattr(market, "log_batches"):
            legacy_logs = market.__dict__.pop("logs", [])
            market.log_batches = [market.pack_logs(legacy_logs[i:i + market.n])
                                  for i in range(0, len(legacy_logs), market.n)]
        quote_minutes = getattr(market, "quote_minutes", 15)
        stride = quote_minutes // QUOTE_MINUTES
        old_ticks_per_day = TICKS_PER_DAY // stride
        if market.phase == old_ticks_per_day:
            market.tick = market.quote_tick // old_ticks_per_day
            market.phase = 0
            if market.history[-1]["狀態"] == "盤中":
                market.history[-1]["狀態"] = "中斷"
        if not hasattr(market, "fill_batches"):
            market.fill_batches = {}
        for quote in market.quotes:
            if "報價序號" not in quote:
                elapsed = (date.fromisoformat(quote["交易日"]) - market.start_date).days
                weeks, remainder = divmod(elapsed, 7)
                weekdays = weeks * 5 + sum((market.start_date.weekday() + i) % 7 < 5 for i in range(remainder))
                hour, minute = map(int, quote["時間"].split(":"))
                quote["報價序號"] = weekdays * old_ticks_per_day + ((hour - 9) * 60 + minute) // quote_minutes
            if stride != 1:
                quote.setdefault("原成交序號", quote["報價序號"])
                quote["報價序號"] *= stride
        if stride != 1:
            market.quote_tick *= stride
            market.phase *= stride
            if hasattr(market, "event_quote_tick"):
                market.event_quote_tick *= stride
            market.fill_batches = {tick * stride: batch for tick, batch in market.fill_batches.items()}
        market.quote_minutes = QUOTE_MINUTES
        if not hasattr(market, "activity_rng"):
            market.activity_rng = random.Random(market.seed + 11000003)
        if not hasattr(market, "institutions"):
            market.institutions = InstitutionBook(market.n, market.seed, market.stock.initial_price, market.stock.taiwan_weight_pct)
        else:
            market.institutions.__class__ = InstitutionBook
            if not hasattr(market.institutions, "execution_plans"):
                market.institutions.execution_plans = {}
            market.institutions.upgrade_initial_capital(market.stock.initial_price, market.stock.taiwan_weight_pct)
        if not hasattr(market, "retail_initial_shares"):
            market.retail_initial_shares = market.n * AI_INITIAL_SHARES
        if not hasattr(market, "retail_initial_cash"):
            old_cash = max(10000.0, market.stock.initial_price * 100)
            cash_per_agent, remainder = divmod(RETAIL_INITIAL_CASH, market.n)
            for i, agent in enumerate(market.agents):
                agent.cash = round(max(0, agent.cash + cash_per_agent + (i < remainder) - old_cash), 2)
            market.retail_initial_cash = RETAIL_INITIAL_CASH
        for agent in market.agents:
            agent.__class__ = Agent
            if "cost_basis" not in vars(agent):
                agent.cost_basis = market.price
                agent.peak_wealth = agent.wealth(market.price)
        if "research_engine" not in vars(market):
            market.research_engine = None
            market.return_variance = 0.0001
        elif market.research_engine is not None:
            # Session State can retain instances of the pre-reload class.
            # Preserve learned arrays/RNGs while binding the fixed methods.
            market.research_engine.__class__ = ResearchEngine
        if "last_explanation" not in vars(market):
            market.last_explanation = None
        if market.event:
            market.event.__class__ = Event
        market.configure_model(config)
        market.configure_institutions(institutional_config)
        if market.stock.symbol in known:
            market.stock = replace(known[market.stock.symbol], initial_price=market.stock.initial_price)
        else:
            old = market.stock
            code = {"AF06": "7827", "AF10": "6806"}.get(old.symbol, "")
            market.stock = Stock(old.symbol, old.name + "（舊模擬）", old.sector, old.initial_price,
                                 old.volatility, old.liquidity, code=code, exchange="舊模擬",
                                 tag="僅供舊部位賣出", active=False)
        if not market.stock.active:
            market.configure_institutions(replace(institutional_config, enabled=False))
    count = first.quote_tick
    additions = []
    newly_listed = set()
    for stock in INSTRUMENTS:
        if stock.symbol not in world.markets:
            market = Market(first.n, world.seed + (int(stock.symbol[2:]) - 1) * 1009,
                            stock, world.volatility, first.start_date, config, institutional_config)
            world.markets[stock.symbol] = market
            if (stock.tag == EXTREME_VOLATILITY_TAG or stock.symbol in NEW_ETF_SYMBOLS
                    or stock.symbol in INDUSTRY_EXPANSION):
                # List new instruments at the current clock instead of fabricating prior trades.
                market.tick, market.quote_tick, market.phase = first.tick, first.quote_tick, first.phase
                market.trading_date = first.trading_date
                if market.phase:
                    market.history.append(market._record("盤中"))
                newly_listed.add(stock.symbol)
            else:
                additions.append(market)
    historical_rng = random.Random(world.seed + 3000003)
    extra_rng = random.Random(world.seed + 4000003)
    new_sectors = list(dict.fromkeys(m.stock.sector for m in additions))
    for quote_index in range(count if additions else 0):
        common = historical_rng.gauss(0, 0.009)
        sectors = {sector: historical_rng.gauss(0, 0.008) for sector in old_sectors}
        sectors.update({sector: extra_rng.gauss(0, 0.008) for sector in new_sectors if sector not in sectors})
        for market in additions:
            market.quote(common, sectors[market.stock.sector],
                         basket_value=world.basket_value(market.stock, quote_index),
                         basket_previous_close=world.basket_value(
                             market.stock, quote_index - quote_index % TICKS_PER_DAY - 1))
    for order in world.player.orders:
        order.__class__ = Order
        if "filled_quantity" not in vars(order) and order.status == "已成交":
            order.filled_quantity = order.quantity
            order.filled_value = round(order.quantity * order.fill_price, 2)
        order.order_type = vars(order).get("order_type", "市價")
        order.limit_price = vars(order).get("limit_price", None)
        market = world.markets[order.symbol]
        next_day = (market.start_date if market.tick == 0 else next_weekday(market.trading_date)) if market.phase == 0 else market.trading_date
        order.valid_day = vars(order).get("valid_day", next_day.isoformat())
        if not market.stock.active and order.side == "買入" and order.status == "待成交":
            order.status = "已取消"
    world.model_config = config
    world.institution_config = next(iter(world.markets.values())).institution_config
    if not hasattr(world, "policy_rate_bps"):
        world.initialize_rate_policy()
    elif getattr(world, "quote_minutes", 15) != QUOTE_MINUTES:
        world.rate_pressure_tick *= getattr(world, "quote_minutes", 15) // QUOTE_MINUTES
    world.quote_minutes = QUOTE_MINUTES
    if not hasattr(world, "scenario_rng"):
        world.initialize_scenarios()
    if not hasattr(world, "auto_news_config"):
        world.auto_news_config = AutoNewsConfig()
        world.random_news_rng = random.Random(world.seed + 7000003)
        for _ in range(first.quote_tick * 8):
            world.random_news_rng.random()
        world.news_feed = []
    weights = ({code: old_weights.get(code, 0) * 100 for code in world.markets}
               if all(code in old_weights or code in newly_listed or market.stock.basket
                      for code, market in world.markets.items())
               else {code: INDEX_WEIGHTS.get(code, 0.0) for code in world.markets})
    if len({market.quote_tick for market in world.markets.values()}) > 1:
        # A live traceback can leave different-length histories. Bind all new
        # methods first, finish the interrupted print, then rebuild the index.
        world.index = WeightedIndex(world.markets, weights, base_points=getattr(getattr(world, "index", None), "base_points", 10000.0))
        world.quote()
    else:
        world.reweight_index(weights)
    world.schema_version = MODEL_VERSION
    return world

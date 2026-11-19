"""Offline, paper-inspired policies. These are adaptations, not LLM replicas.

Literature, equations and deviations are documented in docs/MODEL_RESEARCH.md.
Model randomness is isolated from market shocks and legacy decision draws.
"""
from dataclasses import dataclass, replace, asdict
import math

import numpy as np


PRESETS = {
    "hybrid": "研究混合模型",
    "legacy": "原始規則基準",
    "santa_fe": "Santa Fe 適應性預測",
    "degroot": "DeGroot 社會學習",
    "prospect": "前景理論決策",
    "roles": "角色分析與風控（離線）",
    "zi": "零智慧限價基準",
}


@dataclass(frozen=True)
class ModelConfig:
    mode: str = "legacy"
    use_santa_fe: bool = False
    use_degroot: bool = False
    use_prospect: bool = False
    use_stockagent: bool = False
    use_tradingagents: bool = False
    social_strength: float = 0.35
    self_weight: float = 0.5
    loss_aversion: float = 2.25
    zi_fraction: float = 0.0
    prospect_fraction: float = 0.4
    error_alpha: float = 1 / 75
    evolve_every: int = 250
    exogenous_scale: float = 1.0
    max_exposure: float = 0.65
    benchmark_exposure: float = 0.17
    risk_aversion: float = 40.0

    def __post_init__(self):
        if self.mode not in PRESETS:
            raise ValueError("Unknown model preset")
        for field in ("use_santa_fe", "use_degroot", "use_prospect", "use_stockagent", "use_tradingagents"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be boolean")
        bounds = {"social_strength": (0, 1), "self_weight": (0, 1),
                  "loss_aversion": (0.1, 10), "zi_fraction": (0, 1), "prospect_fraction": (0, 1),
                  "error_alpha": (0.00001, 1), "exogenous_scale": (0, 2),
                  "max_exposure": (0.01, 1), "benchmark_exposure": (0, 1), "risk_aversion": (0.1, 1000)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid {key}")
        if isinstance(self.evolve_every, bool) or not isinstance(self.evolve_every, int) or self.evolve_every < 1:
            raise ValueError("evolve_every must be a positive integer")

    @classmethod
    def preset(cls, mode="hybrid", **overrides):
        if mode not in PRESETS:
            raise ValueError("Unknown model preset")
        fields = dict(mode=mode, use_santa_fe=mode in ("hybrid", "santa_fe"),
                      use_degroot=mode in ("hybrid", "degroot"),
                      use_prospect=mode in ("hybrid", "prospect"),
                      use_stockagent=mode in ("hybrid", "roles"),
                      use_tradingagents=mode in ("hybrid", "roles"),
                      prospect_fraction=1.0 if mode == "prospect" else 0.4,
                      zi_fraction=1.0 if mode == "zi" else 0.15 if mode == "hybrid" else 0.0)
        fields.update(overrides)
        return cls(**fields)

    def to_dict(self):
        return asdict(self)


def degroot_matrix(n, self_weight=0.5):
    """Fixed connected ring with nonnegative, row-stochastic trust weights."""
    if n < 1 or not 0 <= self_weight <= 1:
        raise ValueError("Invalid social graph")
    matrix = np.zeros((n, n))
    for i in range(n):
        peers = sorted({(i - 2) % n, (i - 1) % n, (i + 1) % n, (i + 2) % n} - {i})
        if not peers:
            matrix[i, i] = 1
        else:
            matrix[i, i] = self_weight
            matrix[i, peers] = (1 - self_weight) / len(peers)
    return matrix


def degroot_step(opinions, weights):
    values, matrix = np.asarray(opinions, dtype=float), np.asarray(weights, dtype=float)
    if matrix.shape != (len(values), len(values)) or np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid trust matrix")
    if not np.allclose(matrix.sum(axis=1), 1) or not np.all(np.isfinite(values)):
        raise ValueError("Trust rows must sum to one; beliefs must be finite")
    return matrix @ values


def prospect_value(outcome, loss_aversion=2.25, alpha=0.88, beta=0.88):
    x = np.asarray(outcome, dtype=float)
    return np.where(x >= 0, np.maximum(x, 0) ** alpha,
                    -loss_aversion * np.maximum(-x, 0) ** beta)


def probability_weight(probability, gamma=0.61):
    p = np.asarray(probability, dtype=float)
    if gamma <= 0 or np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("Probability must lie in [0,1]")
    return p ** gamma / (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)


def cumulative_prospect(low, high, probability_up, loss_aversion=2.25):
    """Two-outcome CPT with separate gain/loss cumulative weights (TK 1992)."""
    low, high, p = np.broadcast_arrays(np.asarray(low), np.asarray(high), np.asarray(probability_up))
    if np.any(low > high):
        raise ValueError("Outcomes must be ordered")
    up, down = probability_weight(p, 0.61), probability_weight(1 - p, 0.69)
    vl, vh = prospect_value(low, loss_aversion), prospect_value(high, loss_aversion)
    mixed = down * vl + up * vh
    gains = (1 - up) * vl + up * vh
    losses = down * vl + (1 - down) * vh
    return np.where(low >= 0, gains, np.where(high <= 0, losses, mixed))


@dataclass(frozen=True)
class Observation:
    price: float
    initial_price: float
    fundamental: float
    momentum: float
    short_mean: float
    long_mean: float
    variance: float
    lower: float
    upper: float
    quote_tick: int


@dataclass
class Decisions:
    quantities: np.ndarray
    limits: np.ndarray
    scores: np.ndarray
    reasons: list[str]


class ResearchEngine:
    RULES = 6
    PERSONALITIES = ("保守", "激進", "均衡", "成長")

    def __init__(self, agents, seed, initial_price, config):
        self.config, self.n = config, len(agents)
        self.learning_rng = np.random.default_rng(seed + 5000003)
        self.decision_rng = np.random.default_rng(seed + 6000003)
        rng, shape = self.learning_rng, (self.n, self.RULES)
        self.a = rng.normal(1, 0.012, shape)
        self.b = rng.normal(0, 0.008, shape)
        self.conditions = rng.choice([-1, 0, 1], (*shape, 4), p=[0.8, 0.1, 0.1])
        self.conditions[:, 0, :] = -1  # The default predictor always matches.
        self.a[:, 0], self.b[:, 0] = 1, 0
        self.errors = np.full(shape, 0.0004)
        self.last_predictions = np.zeros(shape)
        self.last_active = np.zeros(shape, dtype=bool)
        self.beliefs = np.zeros(self.n)
        self.last_beliefs = np.zeros(self.n)
        self.news_memory = np.zeros(self.n)
        self.personality = np.arange(self.n) % 4
        self.zi_rank = np.argsort(rng.permutation(self.n))
        self.prospect_rank = np.roll(self.zi_rank, max(1, self.n // 3))
        self.private_value = initial_price * rng.uniform(0.85, 1.15, self.n)
        self.private_cost = initial_price * rng.uniform(0.85, 1.15, self.n)
        self.sensor_bias = rng.normal(0, 0.025, self.n)
        self.graph = degroot_matrix(self.n, config.self_weight)
        self.generations = 0
        self.forecast_mse = None
        self.last_reports = {}
        self.zi_mask = np.zeros(self.n, dtype=bool)
        for i, agent in enumerate(agents):
            agent.personality = self.PERSONALITIES[self.personality[i]]
            if agent.cost_basis <= 0:
                agent.cost_basis = initial_price
            if agent.peak_wealth <= 0:
                agent.peak_wealth = agent.wealth(initial_price)

    def configure(self, config):
        if config.self_weight != self.config.self_weight:
            self.graph = degroot_matrix(self.n, config.self_weight)
        self.config = config

    def forecast(self, observation, news):
        o = observation
        fundamental = o.fundamental * np.exp(self.sensor_bias)
        states = np.column_stack((np.full(self.n, o.momentum > 0),
                                  np.full(self.n, o.price > o.long_mean),
                                  fundamental > o.price, news > 0)).astype(int)
        active = np.all((self.conditions == -1) | (self.conditions == states[:, None, :]), axis=2)
        predictions = np.clip(self.a + self.b * fundamental[:, None] / o.price - 1, -0.1, 0.1)
        complexity = np.sum(self.conditions != -1, axis=2) * 0.000001
        ranking = np.where(active, self.errors + complexity, np.inf)
        best = np.argsort(ranking, axis=1)[:, :2]
        variances = np.take_along_axis(self.errors, best, axis=1)
        weights = np.where(np.take_along_axis(active, best, axis=1), 1 / np.maximum(variances, 1e-6), 0)
        selected = np.take_along_axis(predictions, best, axis=1)
        forecast = np.sum(selected * weights, axis=1) / weights.sum(axis=1)
        variance = np.sum(variances * weights, axis=1) / weights.sum(axis=1)
        self.last_predictions, self.last_active = predictions, active
        return forecast, variance

    def decide(self, agents, observation, base_scores, news):
        o, c, n = observation, self.config, self.n
        # Old sessions may already contain shorts created by the former
        # negative buy-capacity bug. Close those AI positions at the observed
        # price, charging cash rather than gifting shares or resetting wealth.
        if not hasattr(self, "inventory_repairs"):
            self.inventory_repairs = []
        for agent in agents:
            if agent.shares < 0:
                old_shares, old_cash = agent.shares, agent.cash
                agent.cash = round(old_cash + old_shares * o.price, 2)
                agent.shares = 0
                agent.cost_basis = o.price
                self.inventory_repairs.append({
                    "報價序號": o.quote_tick, "代理人": agent.ident,
                    "修復前持股": old_shares, "修復前現金": old_cash,
                    "修復價": o.price, "修復後現金": agent.cash,
                    "方式": "舊版負持股依當前價補回；不計一般成交量",
                })
        news = np.asarray(news, dtype=float)
        # Fixed-sized draws even when a mechanism is disabled: A/B streams align.
        side_roll = self.decision_rng.random(n)
        quantity_roll = self.decision_rng.integers(1, 6, n)
        price_roll = self.decision_rng.random(n)
        sf, variance = self.forecast(o, news)
        private = np.clip(np.asarray(base_scores) * 0.012, -0.1, 0.1)
        self.news_memory = 0.85 * self.news_memory + 0.15 * news
        if c.use_stockagent:
            private = private * np.array([0.8, 1.15, 1.0, 1.08])[self.personality] + 0.003 * self.news_memory
        if c.use_santa_fe:
            private = 0.6 * sf + 0.4 * private
        fundamental = np.clip((o.fundamental * np.exp(self.sensor_bias) / o.price - 1) * 0.08, -0.05, 0.05)
        technical = np.full(n, np.clip((o.short_mean / o.long_mean - 1) * 0.6 + o.momentum * 0.3, -0.05, 0.05))
        message = np.clip(news * 0.018, -0.05, 0.05)
        reports = np.column_stack((fundamental, technical, message))
        bull, bear = reports.max(axis=1), reports.min(axis=1)
        committee = 0.5 * reports.mean(axis=1) + 0.25 * bull + 0.25 * bear
        if c.use_tradingagents:
            private = 0.5 * private + 0.5 * committee
        # Synchronous W*b(t), then private-information anchoring: not pure DeGroot.
        peer = self.graph @ self.beliefs
        belief = (1 - c.social_strength) * private + c.social_strength * peer if c.use_degroot else private
        belief = np.clip(belief, -0.1, 0.1)
        self.beliefs = self.last_beliefs = belief.copy()
        scores = belief / (np.sqrt(np.maximum(variance, 0.000025)) if c.use_santa_fe else 0.015)
        cash = np.array([a.cash for a in agents])
        # A tiny negative floating-point cash residue must never create a
        # negative buy candidate (and hence an unintended short position).
        available_cash = np.maximum(cash, 0)
        shares = np.array([a.shares for a in agents], dtype=int)
        wealth = cash + shares * o.price
        basis = np.array([a.cost_basis for a in agents])
        buy_capacity = np.minimum(5, np.floor(available_cash / o.upper).astype(int))
        sell_capacity = np.minimum(5, shares)
        # AFS inventory anchor: a zero forecast retains benchmark exposure.
        # This bounded portfolio rule is not Santa Fe equilibrium clearing.
        baseline = np.full(n, c.benchmark_exposure)
        if c.use_stockagent:
            baseline = np.array([0.10, 0.30, 0.17, 0.23])[self.personality]
        confidence_variance = variance if c.use_santa_fe else max(o.variance, 0.000025)
        target_fraction = np.clip(baseline + belief / (c.risk_aversion * np.maximum(confidence_variance, 0.000025)), 0, 1)
        target_shares = np.floor(wealth * target_fraction / o.price).astype(int)
        quantities = np.clip(target_shares - shares, -sell_capacity, buy_capacity)
        prospect_mask = self.prospect_rank < int(round(n * c.prospect_fraction)) if c.use_prospect else np.zeros(n, dtype=bool)
        if c.use_prospect:
            candidates = np.column_stack((np.zeros(n, dtype=int), buy_capacity, -sell_capacity))
            sigma = max(0.005, math.sqrt(max(o.variance, 0)))
            p_up = np.clip(0.5 + belief / (2 * sigma), 0.02, 0.98)[:, None]
            if not (0 < o.lower <= o.upper and math.isfinite(o.lower) and math.isfinite(o.upper)):
                raise ValueError("Invalid prospect price bounds")
            # Clip BOTH projected prices on BOTH sides. One-sided clipping
            # reverses the interval when an old observation lies outside it.
            low_price = float(np.clip(o.price * (1 - sigma), o.lower, o.upper))
            high_price = float(np.clip(o.price * (1 + sigma), o.lower, o.upper))
            remaining_shares = shares[:, None] + candidates
            denominator = np.maximum(wealth[:, None], 1)
            reference_pnl = shares[:, None] * (o.price - basis[:, None])
            low = (reference_pnl + remaining_shares * (low_price - o.price)) / denominator
            high = low + remaining_shares * (high_price - low_price) / denominator
            utilities = cumulative_prospect(low, high, p_up, c.loss_aversion)
            choices = np.argmax(utilities, axis=1)  # Hold wins exact ties.
            quantities = np.where(prospect_mask, candidates[np.arange(n), choices], quantities)
        risk_clipped = np.zeros(n, dtype=bool)
        exposure = np.full(n, c.max_exposure)
        if c.use_stockagent:
            exposure = np.minimum(exposure, np.array([0.3, 0.9, 0.6, 0.75])[self.personality])
        if c.use_tradingagents:
            peak = np.maximum(np.array([a.peak_wealth for a in agents]), wealth)
            drawdown = 1 - wealth / np.maximum(peak, 1)
            exposure = np.where(drawdown > 0.2, exposure * 0.5, exposure)
            allowed = np.maximum(0, np.floor(exposure * wealth / o.upper).astype(int) - shares)
            risk_clipped = quantities > allowed
            quantities = np.where(quantities > 0, np.minimum(quantities, allowed), quantities)
        self.zi_mask = self.zi_rank < int(round(n * c.zi_fraction))
        scores = np.where(self.zi_mask, 0, scores)
        belief[self.zi_mask] = 0
        self.beliefs = self.last_beliefs = belief.copy()
        limits = np.full(n, np.nan)
        sides = np.where(side_roll < 1 / 3, -1, np.where(side_roll > 2 / 3, 1, 0))
        bid_high = np.minimum(o.upper, self.private_value)
        ask_low = np.maximum(o.lower, self.private_cost)
        bids = np.floor((o.lower + price_roll * np.maximum(0, bid_high - o.lower)) * 100 + 1e-8) / 100
        asks = np.ceil((ask_low + price_roll * np.maximum(0, o.upper - ask_low)) * 100 - 1e-8) / 100
        zi_qty = np.where(sides > 0, quantity_roll, -quantity_roll)
        zi_qty = np.where(sides == 0, 0, zi_qty)
        zi_qty = np.where((sides > 0) & (bid_high < o.lower), 0, zi_qty)
        zi_qty = np.where((sides < 0) & (ask_low > o.upper), 0, zi_qty)
        zi_qty = np.where(zi_qty > 0, np.minimum(zi_qty, np.floor(available_cash / np.maximum(bids, 0.01)).astype(int)),
                          np.maximum(zi_qty, -shares))
        quantities = np.where(self.zi_mask, zi_qty, quantities).astype(int)
        limits[self.zi_mask] = np.where(sides > 0, bids, asks)[self.zi_mask]
        reasons = ["ZI 隨機限價" if self.zi_mask[i] else "角色風控" if risk_clipped[i] else
                   "前景理論效用" if prospect_mask[i] else "社會信念" if c.use_degroot else
                   "適應性預測" if c.use_santa_fe else "角色分析" if c.use_tradingagents else "私人訊號"
                   for i in range(n)]
        self.last_reports = {"基本面": fundamental, "技術面": technical, "消息面": message,
                             "多方": bull, "空方": bear, "共識": committee, "持倉上限": exposure}
        for i, agent in enumerate(agents):
            agent.belief = float(belief[i])
            agent.forecast = o.price * (1 + float(belief[i]))
            agent.model_family = "ZI-C 簡化" if self.zi_mask[i] else PRESETS[c.mode]
        return Decisions(quantities, limits, scores, reasons)

    def observe(self, realized_return, quote_tick):
        """Called only after execution reveals this period's price."""
        squared = (realized_return - self.last_predictions) ** 2
        updated = (1 - self.config.error_alpha) * self.errors + self.config.error_alpha * squared
        self.errors = np.where(self.last_active, np.maximum(updated, 1e-8), self.errors)
        non_zi = ~self.zi_mask
        self.forecast_mse = float(np.mean((realized_return - self.last_beliefs[non_zi]) ** 2)) if np.any(non_zi) else None
        if self.config.use_santa_fe and quote_tick % self.config.evolve_every == 0:
            self.evolve()

    def evolve(self):
        rng, n = self.learning_rng, self.n
        best = np.argsort(self.errors, axis=1)[:, :2]
        # Keep the always-active default rule; replace one other weak predictor.
        worst = np.argmax(self.errors[:, 1:], axis=1) + 1
        rows = np.arange(n)
        mixing = rng.random(n)
        mutate_a, mutate_b = rng.normal(0, 0.002, n), rng.normal(0, 0.002, n)
        bit_parent = rng.random((n, 4)) < 0.5
        bit_mutate = rng.random((n, 4)) < 0.03
        new_bits = rng.choice([-1, 0, 1], (n, 4), p=[0.8, 0.1, 0.1])
        parents_a, parents_b = self.a[rows[:, None], best], self.b[rows[:, None], best]
        children_a = mixing * parents_a[:, 0] + (1 - mixing) * parents_a[:, 1] + mutate_a
        children_b = mixing * parents_b[:, 0] + (1 - mixing) * parents_b[:, 1] + mutate_b
        inherited = np.where(bit_parent, self.conditions[rows, best[:, 0]], self.conditions[rows, best[:, 1]])
        self.a[rows, worst] = np.clip(children_a, 0.8, 1.2)
        self.b[rows, worst] = np.clip(children_b, -0.2, 0.2)
        self.conditions[rows, worst] = np.where(bit_mutate, new_bits, inherited)
        self.errors[rows, worst] = np.maximum(self.errors.mean(axis=1), self.errors[rows, worst])
        self.generations += 1


def ablation_configs(config=None):
    base = config or ModelConfig.preset("hybrid")
    result = {"legacy": ModelConfig.preset("legacy", exogenous_scale=base.exogenous_scale),
              "zi": ModelConfig.preset("zi", exogenous_scale=base.exogenous_scale), "hybrid": base}
    for field, name in (("use_santa_fe", "no_santa_fe"), ("use_degroot", "no_degroot"),
                        ("use_prospect", "no_prospect"), ("use_stockagent", "no_profiles"),
                        ("use_tradingagents", "no_committee")):
        result[name] = replace(base, **{field: False})
    return result

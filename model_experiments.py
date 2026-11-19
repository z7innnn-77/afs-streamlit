"""Paired, multi-seed simulator diagnostics; no claim of real-market accuracy."""
from dataclasses import replace
import math
import hashlib
from pathlib import Path
import platform

import numpy as np

from market import Market, STOCKS, TICKS_PER_DAY
from research_models import ModelConfig, ablation_configs

SOURCE_HASHES = {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
                 for name in ("market.py", "research_models.py", "model_experiments.py", "market_news.py", "institutions.py")}


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def diagnostics(market, burn_in=20):
    history = market.history[burn_in:]
    prices = np.array([row["股價"] for row in history], dtype=float)
    returns = np.diff(prices) / prices[:-1]
    centered = returns - returns.mean()
    variance = float(np.mean(centered ** 2))
    peak = np.maximum.accumulate(prices)
    volumes = np.array([row["成交量"] for row in history[1:]], dtype=float)
    quotes = market.quotes[burn_in * TICKS_PER_DAY:]
    by_date = {row["交易日"]: row for row in history[1:]}
    hit = sum(q["股價"] in (by_date[q["交易日"]]["跌停價"], by_date[q["交易日"]]["漲停價"]) for q in quotes)
    mse = [q["預測MSE"] for q in quotes if q.get("預測MSE") is not None]
    wealth = np.sort(np.array([a.wealth(market.price) for a in market.agents]))
    gini = float(2 * np.dot(np.arange(1, len(wealth) + 1), wealth) / (len(wealth) * wealth.sum()) - (len(wealth) + 1) / len(wealth))
    return {
        "daily_vol_pct": float(np.std(returns) * 100),
        "excess_kurtosis": float(np.mean(centered ** 4) / variance ** 2 - 3) if variance > 1e-16 else None,
        "return_acf1": correlation(returns[:-1], returns[1:]),
        "abs_return_acf1": correlation(np.abs(returns[:-1]), np.abs(returns[1:])),
        "max_drawdown_pct": float(np.max(1 - prices / peak) * 100),
        "mean_daily_volume": float(volumes.mean()),
        "volume_abs_return_corr": correlation(volumes, np.abs(returns)),
        "forecast_mse": float(np.mean(mse)) if mse else None,
        "limit_hit_pct": hit / max(1, len(quotes)) * 100,
        "order_impact_std_pct": float(np.std([q["訂單衝擊 (%)"] for q in quotes])),
        "external_shock_std_pct": float(np.std([q["外部衝擊 (%)"] for q in quotes])),
        "agent_wealth_gini": gini,
        "generations": market.research_engine.generations if market.research_engine else 0,
        "observed_days": len(returns),
    }


def paired_bootstrap(differences, seed=9173, repetitions=2000):
    values = np.asarray(differences, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def run_comparison(symbol="AF01", seeds=(7, 19, 43, 73), days=80, burn_in=20,
                   agents=48, config=None, variants=None, with_news=True):
    if days < 10 or burn_in < 0 or agents < 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least 10 measured days, positive agents and distinct seeds")
    stock = next((s for s in STOCKS if s.symbol == symbol), None)
    if stock is None:
        raise ValueError("Unknown stock")
    configs = variants or ablation_configs(config)
    if "legacy" not in configs:
        raise ValueError("Comparison requires a legacy control")
    scales = {value.exogenous_scale for value in configs.values()}
    if len(scales) != 1:
        raise ValueError("Paired comparisons must use the same external shock scale")
    rows, paths = [], {}
    for seed in seeds:
        control_state = None
        for name, model in configs.items():
            market = Market(n=agents, seed=seed, stock=stock, model_config=model)
            for day in range(days + burn_in):
                if with_news and day in (burn_in + 5, burn_in + days // 2):
                    tone = 1 if day == burn_in + 5 else -1
                    market.announce("固定排程的模擬消息", tone, 75, 75, "未知", "中立")
                market.advance(1)
            state = (market.environment_rng.getstate(), market.rng.getstate(), market.news_rng.getstate())
            if control_state is None:
                control_state = state
            elif control_state != state:
                raise RuntimeError("External/legacy/news random streams diverged")
            rows.append({"symbol": symbol, "code": stock.code, "variant": name, "seed": seed,
                         **diagnostics(market, burn_in)})
            if seed == seeds[0]:
                paths[name] = [{"day": i, "price": row["股價"]} for i, row in enumerate(market.history)]
    summaries, paired = [], []
    metric_keys = [key for key in rows[0] if key not in ("symbol", "code", "variant", "seed")]
    for name in configs:
        group = [row for row in rows if row["variant"] == name]
        summary = {"symbol": symbol, "code": stock.code, "variant": name, "seeds": len(seeds)}
        for key in metric_keys:
            values = [row[key] for row in group if row[key] is not None and math.isfinite(row[key])]
            summary[key] = float(np.mean(values)) if values else None
        summaries.append(summary)
        if name == "legacy":
            continue
        for key in metric_keys:
            differences = []
            for row in group:
                control = next(r for r in rows if r["variant"] == "legacy" and r["seed"] == row["seed"])
                if row[key] is not None and control[key] is not None:
                    differences.append(row[key] - control[key])
            if differences:
                low, high = paired_bootstrap(differences)
                paired.append({"symbol": symbol, "variant": name, "metric": key,
                               "mean_difference_vs_legacy": float(np.mean(differences)),
                               "ci95_low": low, "ci95_high": high, "paired_seeds": len(differences)})
    return {"runs": rows, "summary": summaries, "paired": paired, "paths": paths,
            "metadata": {"symbol": symbol, "seeds": list(seeds), "days": days, "burn_in": burn_in,
                         "agents": agents, "with_news": with_news,
                         "external_streams_aligned": True,
                         "python_version": platform.python_version(), "numpy_version": np.__version__,
                         "source_sha256": SOURCE_HASHES,
                         "configs": {name: value.to_dict() for name, value in configs.items()},
                         "limitations": "Single-asset synthetic diagnostics, bounded returns, market maker, no real-market calibration or LLM inference. CI is a paired-seed bootstrap, not proof of improvement."}}

"""Synthetic news and deterministic explanations of actual simulator inputs."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AutoNewsConfig:
    enabled: bool = False
    probability: float = 2 / 54
    manual_cooldown: int = 4
    analyst_calls: bool = False

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)) or not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Invalid automatic news probability")
        if isinstance(self.manual_cooldown, bool) or not isinstance(self.manual_cooldown, int) or self.manual_cooldown < 0:
            raise ValueError("Invalid manual news cooldown")


POSITIVE = ("新增長約訂單，市場上修出貨預期", "供應瓶頸改善，產能利用預期升溫", "新產品測試進展順利，需求預期改善")
NEGATIVE = ("客戶延後拉貨，短期訂單預期降溫", "原料成本攀升，獲利預期承壓", "出貨進度受阻，市場下修營運預期")
MACRO_POSITIVE = ("市場風險偏好回升，資金流入股票的意願增加", "需求復甦預期升溫，整體投資信心改善",
                  "通膨低於預期，市場下修未來利率預期", "出口數據優於預期，企業獲利展望改善")
MACRO_NEGATIVE = ("市場風險偏好降溫，資金轉趨保守", "供應鏈成本壓力上升，投資人下修營運預期",
                  "通膨高於預期，市場上修未來利率預期", "海外需求走弱，企業下調出口展望")

# Price sensitivity per unit of rate surprise in this simulation.
RATE_SENSITIVITY = {"金融": 0.30, "半導體": -1.20, "AI 與雲端": -1.35,
                    "記憶體": -1.20, "生技醫療": -1.25, "消費": -0.85}



@dataclass(frozen=True)
class MarketScenario:
    headline: str
    opening_move: float
    daily_pressure: float
    sector_exposure: dict[str, float]
    opening_aftershocks: tuple[float, ...] = ()


MARKET_SCENARIOS = {
    "金融海嘯": MarketScenario("信用市場流動性驟降，全球資金轉向避險", -.067, -.027,
                              {"金融": 1.50, "建材營造": 1.30, "消費": 0.75}),
    "關稅危機": MarketScenario("主要市場提高進口關稅，出口企業下修展望", -.038, -.017,
                              {"半導體": 1.40, "電子零組件": 1.35, "製鞋": 1.45, "汽車": 1.35, "消費": 0.55}),
    "戰事升高": MarketScenario("區域衝突擾亂航線與能源供給，市場風險升高", -.047, -.020,
                              {"航運": -0.45, "能源": -0.35, "塑化": 1.25, "金融": 1.10}),
    "重大天災": MarketScenario("強震造成區域停電與部分工廠停工", -.032, -.015,
                              {"半導體": 1.50, "記憶體": 1.35, "建材營造": -0.30, "水泥": -0.25}),
    "原油價格急漲": MarketScenario("產油區供給中斷，原油價格急漲，市場上修運輸與原料成本", -.019, -.009,
                                   {"能源": -1.50, "航運": 1.35, "塑化": 1.40, "化學": 1.15,
                                    "汽車": 1.10, "金融": 0.50}),
    "原油價格急跌": MarketScenario("原油供給增加，油價急跌，市場下修燃料與部分原料成本", .016, .007,
                                   {"能源": -1.45, "航運": 1.30, "塑化": 1.15, "化學": 1.05,
                                    "汽車": 1.10, "金融": 0.50}),
    "新奈米製程突破": MarketScenario("新一代奈米製程試產達標，供應鏈上修需求展望", .032, .015,
                                  {"半導體": 1.65, "電子零組件": 1.25, "記憶體": 1.25, "消費": 0.45}),
    "新技術加速商用": MarketScenario("新一代運算技術進入量產，企業擴大採購與設備投資", .029, .014,
                                  {"AI 與雲端": 1.70, "散熱": 1.50, "通信網路": 1.30, "半導體": 1.30}),
    "黑天鵝：全球金融市場凍結": MarketScenario(
        "跨國銀行接連停付，融資市場凍結，基金被迫拋售資產", -.205, -.044,
        {"金融": 1.25, "建材營造": 1.10, "通信網路": 0.58, "消費": 0.65},
        opening_aftershocks=(-.143, -.116),
    ),
    "黑天鵝：關鍵供應鏈中斷": MarketScenario(
        "關鍵電網與港口遭大規模中斷，晶片出貨與出口訂單急凍", -.125, -.033,
        {"半導體": 1.45, "電子零組件": 1.25, "記憶體": 1.25, "航運": 1.20, "金融": 0.65},
        opening_aftershocks=(-.090, -.060),
    ),
    "極端利多：全球聯合救市": MarketScenario(
        "主要央行同步推出緊急購債與流動性方案，空頭大舉回補", .171, .035,
        {"金融": 1.20, "建材營造": 1.10, "通信網路": 0.65, "消費": 0.70},
        opening_aftershocks=(.123, .105),
    ),
    "極端利多：科技需求爆發": MarketScenario(
        "關鍵技術提前商用，全球客戶一次釋出多年採購訂單", .100, .030,
        {"半導體": 1.55, "AI 與雲端": 1.60, "散熱": 1.40, "電子零組件": 1.30, "金融": 0.52},
        opening_aftershocks=(.080, .053),
    ),
}
ORDINARY_SCENARIOS = tuple(name for name, event in MARKET_SCENARIOS.items() if not event.opening_aftershocks)
EXTREME_SCENARIOS = tuple(name for name, event in MARKET_SCENARIOS.items() if event.opening_aftershocks)
ORDINARY_SCENARIO_CHANCE = .02
EXTREME_SCENARIO_CHANCE = .0025


def analyst_headline(stock, tone):
    view = "上修評價，喊多" if tone > 0 else "下修評價，喊空"
    return f"【模擬分析師】研究節目{view} {stock.code} {stock.name}"


def draw_news(rng, config, stocks, force=False):
    # Same draw count whether enabled or disabled; no market/agent RNG is used.
    rolls = [rng.random() for _ in range(8)]
    if not stocks or (not force and (not config.enabled or rolls[0] >= config.probability)):
        return None
    stock = stocks[min(len(stocks) - 1, int(rolls[1] * len(stocks)))]
    tone = 1 if rolls[2] >= 0.5 else -1
    if getattr(config, "analyst_calls", False) and rolls[6] < 0.25:
        return {"title": analyst_headline(stock, tone), "tone": tone, "power": 65 + int(rolls[5] * 36),
                "reach": 60 + int(rolls[7] * 41), "truth": "未知", "anchor": "中立",
                "scope": "個股", "stock": stock, "analyst": True}
    scope = "全市場" if rolls[3] < 0.1 else "產業" if rolls[3] < 0.3 else "個股"
    templates = (MACRO_POSITIVE if tone > 0 else MACRO_NEGATIVE) if scope == "全市場" else POSITIVE if tone > 0 else NEGATIVE
    text = templates[min(len(templates) - 1, int(rolls[4] * len(templates)))]
    subject = "全市場" if scope == "全市場" else stock.sector if scope == "產業" else f"{stock.code} {stock.name}"
    return {"title": f"【模擬新聞】{subject}：{text}", "tone": tone,
            "power": 35 + int(rolls[5] * 56), "reach": 45 + int(rolls[6] * 56),
            "truth": "未知", "anchor": "中立", "scope": scope, "stock": stock}


def explain_move(before, after, components, lower, upper, raw_target, active_net, news_title="", news_tone=0, decay=0):
    """Explain recorded price-forming inputs, not an invented news causal effect."""
    actual = (after / before - 1) * 100
    actual_log = math.log(after / before)
    raw_log = sum(components.values())
    residual = actual_log - raw_log
    factor = actual / actual_log if abs(actual_log) > 1e-12 else 100.0
    contributions = {name: value * factor for name, value in components.items()}
    contributions["限制與報價取整"] = residual * factor
    direction = "上漲" if after > before else "下跌" if after < before else "持平"
    same_sign = [(name, value) for name, value in components.items() if value * actual > 0]
    opposing = [(name, value) for name, value in components.items() if value * actual < 0]
    main = max(same_sign, key=lambda item: abs(item[1]))[0] if same_sign else None
    parts = [f"本筆{direction} {actual:+.3f}%（相對上一筆）"]
    if main:
        parts.append(f"主要推力為{main}")
    elif after == before:
        parts.append("合成壓力未形成新的成交價")
    if opposing:
        parts.append(f"{max(opposing, key=lambda item: abs(item[1]))[0]}產生反向抵銷")
    parts.append(f"主動淨買入 {active_net} 股" if active_net >= 0 else f"主動淨賣出 {-active_net} 股")
    bound = ""
    if after == upper and raw_target > upper + 1e-8:
        bound = "漲停限制"
    elif after == lower and raw_target < lower - 1e-8:
        bound = "跌停限制"
    if bound:
        parts.append(f"受到{bound}，超出的價格壓力未反映")
    context = "沒有仍在作用中的新聞。"
    if news_title and decay >= 0.01:
        tone = "利多" if news_tone > 0 else "利空" if news_tone < 0 else "中性"
        context = f"作用中{tone}：{news_title}（剩餘強度 {decay:.0%}）。新聞透過代理人決策影響委託，不代表本筆漲跌必由新聞造成。"
    return {"本筆漲跌 (%)": actual, "漲跌說明": "；".join(parts) + "。",
            "新聞脈絡": context, "價格限制": bound, "分解": contributions}

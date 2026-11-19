"""AFS simulated trading desk: python -m streamlit run app.py."""
from datetime import date, timedelta
from html import escape
from pathlib import Path
import csv
from io import BytesIO, TextIOWrapper
import hashlib
import json
import time

import pandas as pd
import streamlit as st

from execution import BROKER_FEE_RATE, fee, preview_sweep, sales_tax, tick_cents
from charts import candle_chart, index_technicals, intraday_chart, UP, DOWN, FLAT
from market import Exchange, STOCKS, ETFS, CATALOG, MARKET_CLOSE, MODEL_VERSION, ACTIVE_ETF_TAG, EXTREME_VOLATILITY_TAG, INDUSTRY_EXPANSION, MAX_DEPOSIT, QUOTE_MINUTES, TICKS_PER_DAY, InstitutionConfig, upgrade_exchange
from research_models import ModelConfig, PRESETS
from model_experiments import run_comparison
from market_news import AutoNewsConfig, EXTREME_SCENARIO_CHANCE, MARKET_SCENARIOS, ORDINARY_SCENARIO_CHANCE
from record_csv import OHLC, historical_daily_prices, orders_csv, read_history_csv, resume_price_history

st.set_page_config(page_title="AFS | 模擬交易所", page_icon="📈", layout="wide")
mobile = st.toggle("手機版面", key="mobile", bind="query-params",
                   help="將下單與 K 線放在頁面上方，方便用直向螢幕操作。")


def new_world(volatility=1.5, model_config=None, institution_config=None):
    start_date = date.fromisoformat(MARKET_CLOSE["date"]) + timedelta(days=1)
    world = Exchange(volatility=volatility, start_date=start_date, index_base=MARKET_CLOSE["taiex_close"],
                     model_config=model_config or ModelConfig.preset("hybrid"),
                     institution_config=institution_config or InstitutionConfig(enabled=True))
    world.enable_realistic_execution()
    return world


def logs_csv(world):
    rows = world.iter_records("logs")
    first = next(rows, None)
    if first is None:
        return "\n".encode("utf-8-sig")
    buffer = BytesIO()
    with TextIOWrapper(buffer, encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(first), lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)
        output.flush()
        return buffer.getvalue()


def reset_world():
    config = getattr(st.session_state.get("exchange"), "model_config", ModelConfig.preset("hybrid"))
    institutional_config = getattr(st.session_state.get("exchange"), "institution_config", InstitutionConfig(enabled=True))
    st.session_state.exchange = new_world(st.session_state.get("volatility", 1.5), model_config=config,
                                          institution_config=institutional_config)
    st.session_state.comparison = None
    st.session_state.playing = False
    st.session_state.exports = None
    st.session_state.chart_views = {}
    st.session_state.pop("index_weights_editor", None)
    st.session_state.pop("exchange_before_history", None)
    st.session_state.research_comparison = None


def follow_latest_candles():
    st.session_state.review_chart = False
    st.session_state.chart_views = {}
    st.session_state.candle_view_revision = st.session_state.get("candle_view_revision", 0) + 1


def render_candles(history, view_key, timestamp, *, fixed, alt, lookback=60, price_title="價格（元）"):
    views = st.session_state.chart_views
    if not fixed or view_key not in views or views[view_key]["spec"] is None:
        views[view_key] = {"spec": candle_chart(history, lookback, price_title), "time": timestamp}
    view = views[view_key]
    spec = view["spec"]
    if spec is None:
        return False
    latest_date = spec["xAxis"][1]["data"][-1]
    revision = st.session_state.get("candle_view_revision", 0)
    mode = "fixed" if fixed else f"live_{latest_date}"
    st.echarts_chart(spec, height=530, key=f"candles_{view_key}_{mode}_{revision}", alt=alt)
    if fixed:
        st.caption(f"歷史視角固定於 {view['time']}｜目前行情 {timestamp}。點左側「回到最新 K 線」可接續。")
    else:
        st.caption(f"最新 K 線交易日 {latest_date}｜行情時間 {timestamp}")
    return True


MODEL_SWITCHES = ("use_santa_fe", "use_degroot", "use_prospect", "use_stockagent", "use_tradingagents")
MODEL_VALUES = ("social_strength", "loss_aversion", "zi_fraction", "prospect_fraction", "exogenous_scale", "evolve_every")


def populate_model_form(config):
    for field in MODEL_SWITCHES + MODEL_VALUES:
        st.session_state[f"model_{field}"] = getattr(config, field)


def change_model_preset():
    populate_model_form(ModelConfig.preset(st.session_state.model_preset))


@st.cache_data(max_entries=4, show_spinner=False)
def cached_model_comparison(symbol, seeds, days, config_json, source_signature):
    return run_comparison(symbol, seeds=seeds, days=days, agents=32,
                          config=ModelConfig(**json.loads(config_json)))


def direction_style(value):
    color = UP if value > 0 else DOWN if value < 0 else FLAT
    return f"color: {color}"


def styled_prices(frame, columns):
    return frame.style.format({name: "{:,.2f}" for name in frame.select_dtypes("number").columns}).map(
        direction_style, subset=columns)


if "exchange" not in st.session_state:
    st.session_state.exchange = new_world()
    st.session_state.comparison = None
    st.session_state.playing = False
    st.session_state.exports = None
elif getattr(st.session_state.exchange, "schema_version", None) != MODEL_VERSION:
    if all(hasattr(m, "quotes") for m in st.session_state.exchange.markets.values()):
        previous_version = getattr(st.session_state.exchange, "schema_version", None)
        st.session_state.exchange = upgrade_exchange(st.session_state.exchange)
        st.session_state.exchange.enable_realistic_execution()
        st.session_state.chart_views = {}
        st.session_state.exports = None
        st.session_state.pop("index_weights_editor", None)
        if previous_version == 20:
            st.info("已啟用五檔、部分成交與交易成本，並接續既有帳戶和行情。")
        elif previous_version == 15:
            st.info("已更新權值股設定，外資持股更集中，一般委託的價格影響較小。")
        elif previous_version == 14:
            st.info("已加入加權指數技術分析，並擴大模擬外資的資金與下單能力。")
        elif previous_version == 13:
            st.info("已更新 ETF 定價，後續報價會以成分籃子的當日漲跌為基準。")
        elif previous_version == 12:
            st.info("已更新法人設定相容性，可繼續使用既有會話。")
        elif previous_version == 11:
            st.info("已更新指數重建，舊報價缺漏會依時間對齊。")
        elif previous_version == 10:
            st.info("已加入 ETF、每日排行與雙邊成交明細。")
        elif previous_version == 9:
            st.info("已加入模擬入金，可在「我的下單台」增加資金。")
        else:
            st.info("已更新研究引擎並接續既有會話；保留玩家現金、持股與行情。舊 AI 負持股會依模擬現價修復，中斷報價會先補齊。")
    else:
        reset_world()
        st.info("已更新為盤中交易模型並建立新的模擬世界。")
st.session_state.setdefault("comparison", None)
st.session_state.setdefault("exports", None)
st.session_state.setdefault("chart_views", {})
st.session_state.setdefault("research_comparison", None)
history_action = st.session_state.pop("history_exchange_action", None)
if history_action:
    try:
        if history_action == "resume":
            current = st.session_state.exchange
            _, prices = st.session_state.historical_tables["行情"]
            history = resume_price_history(prices, current.markets)
            resumed = new_world(current.volatility, current.model_config, current.institution_config)
            resumed.seed_daily_history(history)
            st.session_state.setdefault("exchange_before_history", current)
            st.session_state.exchange = resumed
        else:
            st.session_state.exchange = st.session_state.pop("exchange_before_history")
    except ValueError as error:
        st.error(str(error))
    else:
        follow_latest_candles()
        st.session_state.playing = False
        st.session_state.exports = None
        st.session_state.pop("index_weights_editor", None)
exchange = st.session_state.exchange
if "model_preset" not in st.session_state:
    st.session_state.model_preset = exchange.model_config.mode
    populate_model_form(exchange.model_config)
stock_labels = {code: f"{m.stock.code or code} · {m.stock.name}" for code, m in exchange.markets.items()}
active_codes = [code for code, m in exchange.markets.items() if m.stock.active]
INDEX_VIEW = "index"

st.title("📈 AFS 模擬交易所")
st.caption(f"{len(active_codes)} 檔台股模擬標的（含 {len(ETFS)} 檔 ETF）｜每日 ±10%")
if getattr(exchange.index, "base_points", None) == MARKET_CLOSE["taiex_close"]:
    st.caption(f"{MARKET_CLOSE['date']} 官方收盤為起點｜加權指數基期 {MARKET_CLOSE['taiex_close']:,.2f} 點")
else:
    st.caption("目前是既有模擬。另開遊戲分頁可從 11/02 官方收盤價開始。")

with st.sidebar:
    st.header("行情控制")
    universe_filter = st.selectbox("股票分類", ["全部", "ETF", ACTIVE_ETF_TAG, "一般ETF", "官方可當沖", "塑化／化學", "新增產業股", "高波動題材", EXTREME_VOLATILITY_TAG, "冷門低量", "新增：國巨／華通／全景軟體", "舊模擬部位"], key="universe_filter")
    options = [code for code, m in exchange.markets.items() if
               (universe_filter == "全部" and m.stock.active) or
               (universe_filter == "ETF" and bool(m.stock.basket)) or
               (universe_filter == ACTIVE_ETF_TAG and m.stock.tag == ACTIVE_ETF_TAG) or
               (universe_filter == "一般ETF" and bool(m.stock.basket) and m.stock.tag != ACTIVE_ETF_TAG) or
               (universe_filter == "官方可當沖" and m.stock.active and m.stock.day_trade_verified) or
               (universe_filter == "塑化／化學" and m.stock.sector in ("塑化", "化學")) or
               (universe_filter == "新增產業股" and code in INDUSTRY_EXPANSION) or
               (universe_filter == "高波動題材" and m.stock.active and m.stock.tag == "高波動題材") or
               (universe_filter == EXTREME_VOLATILITY_TAG and m.stock.active and m.stock.tag == EXTREME_VOLATILITY_TAG) or
               (universe_filter == "冷門低量" and m.stock.active and m.thinly_traded) or
               (universe_filter == "新增：國巨／華通／全景軟體" and m.stock.code in ("2327", "2313", "8272")) or
               (universe_filter == "舊模擬部位" and not m.stock.active)]
    if not options:
        st.caption("此分類目前沒有標的，先顯示全部股票。")
        options = active_codes
    default_symbol = options[0]
    if universe_filter == "全部":
        options.insert(0, INDEX_VIEW)
    if st.session_state.get("stock") not in options:
        st.session_state.stock = default_symbol
    symbol = st.selectbox("觀察股票", options,
                          format_func=lambda code: "加權指數" if code == INDEX_VIEW else stock_labels[code],
                          key="stock")
    watching_index = symbol == INDEX_VIEW
    if universe_filter == EXTREME_VOLATILITY_TAG:
        st.caption("此分類的模擬股較容易連續急漲或急跌，掛單深度也較薄。")
    playing = st.toggle("自動跳動行情", value=False, key="playing")
    interval = st.slider("每筆報價間隔（秒）", 0.5, 3.0, 1.0, 0.5, key="quote_interval")
    exchange.volatility = st.slider("市場波動強度", 0.5, 3.0, 1.5, 0.1, key="volatility")
    review_chart = st.toggle("固定 K 線，拖曳看歷史", value=False, key="review_chart",
                             help="固定圖表後可持續拖曳與縮放，行情和下單繼續運作；關閉即回到最新 K 線。")
    st.button("回到最新 K 線", key="follow_latest_candles", on_click=follow_latest_candles, width="stretch")
    st.caption(f"每 {QUOTE_MINUTES} 分鐘一筆，09:05–13:30 共 {TICKS_PER_DAY} 筆。週末休市。")
    if st.button("下一筆報價", key="next_quote", width="stretch", disabled=playing):
        exchange.quote()
    if st.button("推進至下一次收盤", key="step1", width="stretch", disabled=playing):
        exchange.advance(1)
    if st.button("推進 10 個交易日", key="step10", width="stretch", disabled=playing):
        exchange.advance(10)
    st.button("重置市場與帳戶", key="reset", width="stretch", on_click=reset_world)
    with st.expander("論文模型控制"):
        st.selectbox("引擎預設", list(PRESETS), format_func=PRESETS.get,
                     key="model_preset", on_change=change_model_preset)
        benchmark = st.session_state.model_preset in ("legacy", "zi")
        st.caption("這是離線數值模型；StockAgent／TradingAgents 只借用角色與流程，尚未接入 LLM。選擇後按套用。")
        with st.form("model_config_form"):
            for field, label in (
                ("use_santa_fe", "Santa Fe：預測規則學習與演化"),
                ("use_degroot", "DeGroot：社會信念傳播"),
                ("use_prospect", "前景理論：損失厭惡與機率加權"),
                ("use_stockagent", "StockAgent 啟發：異質人物與消息記憶"),
                ("use_tradingagents", "TradingAgents 啟發：角色分析與風控"),
            ):
                st.checkbox(label, key=f"model_{field}", disabled=benchmark)
            st.slider("社會影響強度", 0.0, 1.0, step=0.05, key="model_social_strength", disabled=benchmark)
            st.slider("損失厭惡 λ", 0.5, 4.0, step=0.05, key="model_loss_aversion", disabled=benchmark)
            st.slider("前景理論族群比例", 0.0, 1.0, step=0.05, key="model_prospect_fraction", disabled=benchmark)
            st.slider("零智慧族群比例", 0.0, 1.0, step=0.05, key="model_zi_fraction", disabled=benchmark)
            st.slider("外部價格衝擊倍率", 0.0, 1.0, step=0.05, key="model_exogenous_scale")
            st.selectbox("規則演化間隔（筆）", [36, 250, 1000], key="model_evolve_every", disabled=benchmark,
                         help="250／1000 參考 Santa Fe 中速／慢速設定；36 是本專案快速教學設定。")
            apply_model = st.form_submit_button("套用模型", key="apply_model", type="primary")
        if apply_model:
            try:
                settings = {field: st.session_state[f"model_{field}"] for field in MODEL_SWITCHES + MODEL_VALUES}
                config = ModelConfig(mode=st.session_state.model_preset,
                                     error_alpha=1 / 150 if settings["evolve_every"] == 1000 else 1 / 75,
                                     **settings)
                exchange.configure_model(config)
                st.success("已套用到後續報價；帳戶與既有行情保留。")
            except ValueError as error:
                st.error(str(error))
    with st.expander("三大法人控制"):
        pure_zi = exchange.model_config.mode == "zi"
        if pure_zi:
            st.session_state.institutions_enabled = False
        enabled = st.toggle("啟用模擬三大法人", value=exchange.institution_config.enabled,
                            key="institutions_enabled", disabled=pure_zi)
        institutional_activity = st.slider("法人交易活躍度", 0.0, 3.0, float(exchange.institution_config.activity),
                                          0.1, key="institution_activity", disabled=pure_zi)
        exchange.configure_institutions(InstitutionConfig(enabled=enabled, activity=institutional_activity))
        st.caption("外資、投信、自營商（自行買賣／避險庫存代理）均有獨立模擬資金與持股，不是真實法人資料。純 ZI 基準停用法人；關閉不會清空法人舊部位。")
    with st.expander("加權指數權重設定"):
        st.caption(f"參考 {CATALOG['weight_asof']} 官方市值比重，在所選上市股內正規化至100%；上櫃預設0。可手動修改，合計需100%。")
        with st.form("index_weight_form"):
            edited_weights = st.data_editor(
                pd.DataFrame([{"代碼": code, "股票": stock_labels[code], "基期權重 (%)": weight * 100}
                              for code, weight in exchange.index.weights.items()]),
                disabled=["代碼", "股票"], hide_index=True, key="index_weights_editor",
                column_config={"基期權重 (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1, required=True)},
                height=400, alt="可調整台股模擬指數各股基期權重",
            )
            apply_weights = st.form_submit_button("套用指數權重", key="apply_weights")
        if apply_weights:
            try:
                exchange.reweight_index(dict(zip(edited_weights["代碼"], edited_weights["基期權重 (%)"])))
                st.session_state.chart_views = {}
                st.success("權重已套用，指數歷史已依新權重重算。")
            except ValueError as error:
                st.error(str(error))
    with st.expander("股票清單與資料來源"):
        st.caption(f"權重參考日：{CATALOG['weight_asof']}｜當沖名單參考日：{CATALOG['day_trade_asof']}。資料是離線快照，不會自動更新資格。")
        st.markdown(f"[期交所市值權重表]({CATALOG['weight_source']}) · [證交所當沖名單](https://www.twse.com.tw/zh/trading/day-trading.html) · [櫃買當沖名單](https://www.tpex.org.tw/zh-tw/mainboard/trading/day-trading/securities.html)")
        st.caption("起始價採官方收盤快照，波動分類是模擬設定。")
    with st.expander("隨機利多／利空新聞", expanded=True):
        random_news = st.toggle("啟用隨機新聞", value=True, key="random_news")
        news_daily = st.slider("平均每日新聞數（全市場合計）", 0, 6, 2, key="news_daily")
        analyst_calls = st.toggle("隨機加入分析師喊盤", value=True, key="analyst_calls")
        exchange.configure_news(AutoNewsConfig(enabled=random_news, probability=news_daily / TICKS_PER_DAY,
                                               analyst_calls=analyst_calls))
        bullish, bearish = st.columns(2)
        for column, direction, key, tone in ((bullish, "喊多", "analyst_buy", 1), (bearish, "喊空", "analyst_sell", -1)):
            if column.button(f"分析師{direction}本股", key=key, disabled=watching_index or not exchange.markets[symbol].stock.active):
                st.success(exchange.publish_analyst_call(symbol, tone)["消息"])
        if st.button("立即產生一則模擬新聞", key="generate_news", width="stretch"):
            news = exchange.generate_random_news(force=True)
            if news:
                st.success(news["消息"])
        st.caption("每筆報價判斷是否發布，數量並非每日固定。自動新聞避開近4筆的手動消息；按立即產生可取代。全部為虛構情境，利多不保證上漲。")
    if not watching_index and exchange.markets[symbol].stock.active:
        with st.expander("模擬目標價新聞"):
            reference = exchange.markets[symbol].price
            st.caption(f"{stock_labels[symbol]}｜目前參考價 {reference:,.2f} 元")
            forecast = st.number_input("預期目標價（元）", min_value=0.01, max_value=1_000_000_000.0,
                                       value=min(1_000_000_000.0, round(reference * 1.05, 2)),
                                       step=0.01, format="%.2f", key=f"forecast_price_{symbol}")
            if st.button("公布目標價", key="publish_price_forecast", width="stretch"):
                try:
                    st.success(exchange.publish_price_forecast(symbol, forecast)["消息"])
                except ValueError as error:
                    st.error(str(error))
            st.caption("目標價高於發布時參考價會形成偏多消息，低於參考價則偏空。可在下一筆報價查看跟單成交。")
    with st.expander("重大市場事件"):
        exchange.auto_scenarios = st.toggle("隨機重大事件", value=True, key="auto_scenarios")
        scenario = st.selectbox("模擬情境", list(MARKET_SCENARIOS), key="market_scenario")
        if st.button("觸發重大事件", key="start_scenario", width="stretch"):
            try:
                st.success(exchange.start_scenario(scenario)["消息"])
            except ValueError as error:
                st.info(str(error))
        st.caption(f"無其他重大事件時，每個開盤有 {ORDINARY_SCENARIO_CHANCE:.2%} 機率觸發一般事件，"
                   f"{EXTREME_SCENARIO_CHANCE:.2%} 機率觸發極端事件。")
        if MARKET_SCENARIOS[scenario].opening_aftershocks:
            st.caption("極端事件會衝擊下一筆報價，接下來兩個交易日開盤還會受到餘波影響。每日漲跌停仍生效。")
    with st.expander("模擬央行利率"):
        auto_rate = st.toggle("啟用隨機利率會議", value=True, key="auto_rate")
        meeting_days = st.slider("會議間隔（交易日）", 5, 60, 20, key="rate_meeting_days")
        exchange.rate_meeting_interval = meeting_days if auto_rate else 0
        size = st.selectbox("手動調整幅度", [25, 50], format_func=lambda bp: f"{bp} 基點", key="rate_change_bp")
        hike, cut = st.columns(2)
        if hike.button("升息", key="rate_hike", width="stretch"):
            st.success(exchange.apply_rate_decision(size)["消息"])
        if cut.button("降息", key="rate_cut", width="stretch"):
            st.success(exchange.apply_rate_decision(-size)["消息"])
        st.caption("利率自 2.00% 起算。會議可升息、降息或維持；手動決議從下一筆報價開始影響。25 基點等於 0.25 個百分點。")
    with st.expander("新聞控制台"):
        target_label = st.selectbox("新聞影響範圍", ["全市場", *active_codes],
                                    format_func=lambda value: stock_labels.get(value, value), key="news_target")
        target = None if target_label == "全市場" else target_label
        headline = st.text_area("新聞內容", "模擬企業宣布突破性新技術", height=90)
        tone_label = st.selectbox("消息方向", ["利多", "利空", "中性"])
        tone = {"利多": 1, "利空": -1, "中性": 0}[tone_label]
        power = st.slider("消息強度", 0, 100, 75)
        reach = st.slider("觸及代理人比例", 0, 100, 75)
        truth = st.radio("消息真相（上帝視角）", ["真實", "虛假", "未知"])
        anchor = st.selectbox("新聞主播播報立場", ["中立", "偏多", "偏空"])
        if st.button("發布新聞", key="publish", type="primary", width="stretch"):
            if headline.strip():
                exchange.announce(headline, tone, power, reach, truth, anchor, target)
                st.success(f"已發布至{stock_labels.get(target, '全市場')}。")
            else:
                st.warning("請先輸入新聞內容。")


def render_market_ticker(world, stock_labels):
    def item(label, value, change):
        color = UP if change > 0 else DOWN if change < 0 else FLAT
        return (f'<span class="afs-ticker-item">{escape(label)} <b style="color:{color}">'
                f'{value:,.2f} {change:+.2f}%</b></span>')

    items = [item("加權指數", world.index.value, world.index.history[-1]["漲跌幅 (%)"])]
    items.extend(item(stock_labels[symbol], market.price, market.history[-1]["漲跌幅 (%)"])
                 for symbol, market in world.markets.items() if market.stock.active)
    headlines = [
        f'<span class="afs-ticker-item">{escape(row["發布時間"])} · {escape(row["消息"])}</span>'
        for row in reversed(world.news_feed[-5:])
    ]

    def strip(label, entries, duration, *, offset=0):
        contents = "".join(entries)
        return f"""
            <div class="afs-ticker" aria-label="{label}">
              <span class="afs-ticker-label">{label}</span>
              <div class="afs-ticker-view" tabindex="0">
                <div class="afs-ticker-track" style="--afs-duration:{duration}s;--afs-delay:{offset}s">
                  <div class="afs-ticker-group">{contents}</div>
                  <div class="afs-ticker-group" aria-hidden="true">{contents}</div>
                </div>
              </div>
            </div>
        """

    duration = max(24, len(items) * 2.6)
    market_strip = strip("AFS 行情台", items, duration, offset=-(time.monotonic() % duration))
    news_duration = max(24, len(headlines) * 12)
    news_strip = strip("新聞快訊", headlines or ['<span class="afs-ticker-item">目前沒有模擬新聞</span>'],
                       news_duration)
    st.html(f"""
        <style>
        .afs-ticker {{display:flex;align-items:stretch;border-radius:.45rem;background:#172031;color:#f0f4fa}}
        .afs-ticker-label {{padding:.48rem .7rem;background:#223a5a;font-weight:700;white-space:nowrap}}
        .afs-ticker-view {{flex:1;min-width:0;overflow:hidden;container-type:inline-size}}
        .afs-ticker-track {{display:flex;width:max-content;animation:afs-ticker-scroll var(--afs-duration) var(--afs-delay) linear infinite}}
        .afs-ticker-view:hover .afs-ticker-track,
        .afs-ticker-view:focus-within .afs-ticker-track {{animation-play-state:paused}}
        .afs-ticker-group {{display:flex;flex:none;min-width:100cqw}}
        .afs-ticker-item {{padding:.48rem .8rem;white-space:nowrap;font-variant-numeric:tabular-nums}}
        @keyframes afs-ticker-scroll {{to {{transform:translateX(-50%)}}}}
        @media (prefers-reduced-motion:reduce) {{
          .afs-ticker-view {{overflow-x:auto}} .afs-ticker-track {{animation:none}}
          .afs-ticker-group[aria-hidden=true] {{display:none}}
        }}
        </style>
        {market_strip}
    """)
    st.html(news_strip)


def render_index_composition(composition):
    st.caption(f"所選上市股合計涵蓋參考日大盤市值 {CATALOG['selected_taiwan_weight_pct']:.4f}%，先在此範圍正規化權重再建立固定股數籃子；上櫃預設0。原始台股參考占比與模擬指數占比分列。")
    st.bar_chart(composition.set_index("股票")["指數占比 (%)"], horizontal=True, height=320,
                 alt="各股票目前在模擬加權指數的占比")
    st.dataframe(styled_prices(composition, ["貢獻點數"]), hide_index=True,
                 alt="各股基期權重、目前指數占比與當日貢獻點數")


@st.fragment(run_every=interval if playing else None)
def trading_desk():
    world = st.session_state.exchange
    world.auto_quote(time.monotonic(), interval, playing)
    render_market_ticker(world, stock_labels)
    imported_days = getattr(world, "imported_days", ())
    if imported_days:
        st.caption(f"行情已接續至 {imported_days[-1]}｜下一筆報價 {world.next_timestamp}")
    status = "盤中" if world.phase else "收盤"
    if world.phase:
        progress = f"第 {world.tick + 1} 個交易日 · {world.phase}/{TICKS_PER_DAY} 筆"
    else:
        progress = f"已完成 {world.tick} 個交易日"
    st.caption(f"{world.timestamp} · {status} · {progress} · {'自動報價中' if playing else '報價已暫停'}")
    active_scenario = world.scenario_at(next(iter(world.markets.values())).quote_tick)
    if active_scenario:
        st.caption(f"進行中的模擬事件｜{active_scenario}")
    st.caption(f"目前引擎：{PRESETS[world.model_config.mode]}｜外部衝擊倍率 {world.model_config.exogenous_scale:.2f}｜所有角色推論均為離線數值規則。")

    if not watching_index:
        if mobile:
            order_area = st.container()
            chart_area = st.container()
        else:
            chart_area, order_area = st.columns([2.5, 1])

    index = world.index
    index_day = index.history[-1]
    index_return = index_day["漲跌幅 (%)"]
    index_color = UP if index_return > 0 else DOWN if index_return < 0 else FLAT
    composition = pd.DataFrame(index.composition(world.markets))
    current_weights = dict(zip(composition["代碼"], composition["指數占比 (%)"]))
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("AFS 加權指數", f"{index.value:,.2f}",
              f"{index.value - index.previous_close:+,.2f} / {index_day['漲跌幅 (%)']:+.2f}%", delta_color="inverse")
    i2.metric("指數昨收", f"{index.previous_close:,.2f}")
    i3.metric("上漲 / 下跌家數", f"{sum(m.history[-1]['漲跌幅 (%)'] > 0 for m in world.markets.values())} / {sum(m.history[-1]['漲跌幅 (%)'] < 0 for m in world.markets.values())}")
    if watching_index:
        i4.metric("全市場成交量（股）", f"{index_day['成交量']:,}")
    else:
        i4.metric(f"{world.markets[symbol].stock.name} · 指數占比", f"{current_weights[symbol]:.2f}%")
    with st.expander("加權指數技術分析", key="index_technicals", on_change="rerun") as technicals:
        if technicals.open:
            indicators = index_technicals(index.history)
            if indicators.empty:
                st.caption("首個交易日收盤後開始累積指標。")
            else:
                st.caption(f"以已收盤的模擬指數計算，最新資料 {indicators.index[-1]}。資料不足的指標會留空。")
                last = indicators.iloc[-1]
                for column, name in zip(st.columns(4), ("MA5", "MA20", "MA60", "RSI14")):
                    column.metric(name, f"{last[name]:,.2f}" if pd.notna(last[name]) else "資料不足")
                st.line_chart(indicators[["收盤", "MA5", "MA20", "MA60"]].tail(120),
                              height=260, alt="加權指數收盤與 5、20、60 日均線")
                if indicators["RSI14"].notna().any():
                    st.line_chart(indicators["RSI14"].tail(120), height=150, alt="加權指數 RSI14")
                if indicators["DIF"].notna().any():
                    st.line_chart(indicators[["DIF", "訊號線"]].tail(120), height=160, alt="加權指數 MACD 線")
                if indicators["MACD 柱"].notna().any():
                    st.bar_chart(indicators["MACD 柱"].tail(120), height=140, alt="加權指數 MACD 柱")
    if watching_index:
        st.subheader("加權指數 · 日 K")
        lookback = st.selectbox("顯示交易日數", [30, 60, 120], index=1, key="index_lookback")
        if not render_candles(index.history, f"index:{lookback}", world.timestamp,
                              fixed=review_chart, lookback=lookback, price_title="指數點數",
                              alt="AFS 加權指數日 K 與全市場成交量"):
            st.caption("推進第一筆報價後開始顯示指數 K 線。")
        intraday = intraday_chart(index.quotes, index_day["昨收"], price_title="指數點數")
        if intraday is not None:
            st.altair_chart(intraday, alt="AFS 加權指數盤中走勢")
        with st.expander("加權指數成分占比"):
            render_index_composition(composition)
        return

    market = world.markets[symbol]
    player = world.player
    latest = market.history[-1]
    st.line_chart(pd.DataFrame(index.history).set_index("交易日")["股價"], height=150,
                  color=UP if index.value >= index.previous_close else DOWN,
                  alt="AFS 模擬加權指數每日走勢")

    p2, p3, p4 = st.columns(3)
    p2.metric("今日最高 / 最低", f"{latest['最高']:,.2f} / {latest['最低']:,.2f}")
    p3.metric("今日成交量（股）", f"{latest['成交量']:,}")
    p4.metric("今日漲停 / 跌停", f"{latest['漲停價']:,.2f} / {latest['跌停價']:,.2f}")

    st.subheader("本筆漲跌原因")
    quote = market.quotes[-1] if market.quotes else {}
    st.info(quote.get("漲跌說明", "這筆為舊版紀錄，下一筆報價開始提供說明。"))
    st.caption(quote.get("新聞脈絡", ""))
    institution_rows = market.institution_snapshot(player.positions[symbol].quantity if symbol in player.positions else 0)
    institution_by_group = {row["法人"]: row for row in institution_rows}
    st.caption(f"模擬三大法人：{'交易啟用' if market.institution_config.enabled else '交易暫停'}｜以下為今日累計，單位：股")
    columns = st.columns(4)
    for column, group in zip(columns[:3], ("外資", "投信", "自營商")):
        net = institution_by_group[group]["買賣超股數"]
        column.metric(f"{group}買賣超", f"{net:+,}", net, delta_color="inverse")
    total_net = sum(row["買賣超股數"] for row in institution_rows)
    columns[3].metric("三大法人合計", f"{total_net:+,}", total_net, delta_color="inverse")
    st.caption(f"模擬政策利率 {world.policy_rate_bps / 100:.2f}%｜科技與生技對升降息較敏感，金融股採較弱的反向反應。")
    if world.news_feed:
        news = world.news_feed[-1]
        st.write(f"**最新市場消息 · {news['發布時間']} · {news['來源']} · {news['影響範圍']}**")
        st.write(news["消息"])

    with chart_area:
        st.html(f"""
            <div style="display:flex;flex-wrap:wrap;justify-content:space-between;
                        align-items:center;gap:.4rem .75rem">
                <h3 style="margin:0;padding:0">{escape(stock_labels[symbol])} · 日 K</h3>
                <div aria-label="模擬加權指數" style="display:inline-flex;align-items:center;
                     gap:.4rem;padding:.15rem .55rem;border:1px solid var(--border-color,#8885);
                     border-radius:1rem;font-size:.8rem;white-space:nowrap;font-variant-numeric:tabular-nums">
                    <span>加權指數</span><strong>{index.value:,.2f}</strong>
                    <strong style="color:{index_color}">{index_return:+.2f}%</strong>
                </div>
            </div>
        """)
        if market.stock.basket:
            holdings = world.etf_holdings_value(market.stock, market.quote_tick - 1)
            nav = sum(value for _, value in holdings)
            components = "、".join(f"{world.markets[code].stock.name} {value / nav:.1%}" for code, value in holdings)
            if market.stock.tag == ACTIVE_ETF_TAG:
                last_adjustment = market.etf_allocation.day.isoformat() if market.etf_allocation else "等待首日收盤"
                st.caption(f"主動式模擬策略｜開盤前依最近最多 5 個交易日表現調整配置｜最近調整 {last_adjustment}")
            else:
                st.caption("被動式模擬策略｜固定成分股股數，權重隨個股價格變動")
            st.caption(f"目前模擬持股占比 {components}")
            premium = (market.price / nav - 1) * 100
            status = "溢價" if premium >= 0.005 else "折價" if premium <= -0.005 else "平價"
            st.caption(f"市價 {market.price:,.2f} 元｜估計淨值 {nav:,.2f} 元｜{status} {premium:+.2f}%")
        lookback = st.selectbox("顯示交易日數", [30, 60, 120], index=1, key="lookback")
        if not render_candles(market.history, f"{symbol}:{lookback}", world.timestamp,
                              fixed=review_chart, lookback=lookback,
                              alt="可拖曳及縮放的日 K 與成交量，紅漲綠跌；下方拉桿選擇歷史日期"):
            st.caption(f"基準收盤 {market.price:,.2f} 元。推進第一筆報價後開始顯示模擬日 K。")
        if mobile:
            st.caption("左右拖動下方滑桿可移動日期範圍，拉動滑桿兩端可縮放。")
        else:
            st.caption("拖曳圖表左右移動、滾輪縮放、下方拉桿選日期，左上角可重設視角。")
        st.caption("紅 K：收盤高於開盤｜綠 K：收盤低於開盤｜灰色十字：平盤｜橙線 MA5、藍線 MA20。最後一根盤中持續更新。")
        intraday = intraday_chart(market.quotes, latest["昨收"])
        if intraday is not None:
            st.altair_chart(intraday, alt="當日逐筆模擬報價與昨收參考線")
    with order_area:
        st.subheader("我的下單台")
        change = market.price - latest["昨收"]
        st.metric(f"{market.stock.code} {market.stock.name} · 模擬參考價", f"{market.price:,.2f} 元",
                  f"{change:+,.2f} / {latest['漲跌幅 (%)']:+.2f}%（較昨收）", delta_color="inverse")
        if quote:
            st.caption(f"報價時間 {quote['交易日']} {quote['時間']}｜較上一筆 {quote['本筆漲跌 (%)']:+.2f}%｜本筆成交金額 {quote.get('成交金額', quote['股價'] * quote['成交量']):,.2f} 元（{quote['成交量']:,} 股）")
        with st.expander("模擬買賣五檔", expanded=True):
            book = market.book if market.phase and market.book is not None else market.indicative_book()
            bids = [(price, size) for price, size in book["買入"] if size]
            asks = [(price, size) for price, size in book["賣出"] if size]
            levels = []
            for i in range(5):
                bid, bid_qty = bids[i] if i < len(bids) else (None, None)
                ask, ask_qty = asks[i] if i < len(asks) else (None, None)
                levels.append({"檔位": i + 1, "買價": bid, "買量（股）": bid_qty, "賣價": ask, "賣量（股）": ask_qty})
            st.dataframe(levels, hide_index=True, alt="模擬委託簿買賣五檔")
            st.caption("盤中顯示上次撮合剩餘深度，收盤時顯示下一時段估算。下一筆報價時價位與數量會更新。")
        with st.expander("電腦與法人持股", on_change="rerun", key="initial_holdings") as holdings:
            if holdings.open:
                balances = [{"交易者": "一般電腦交易者", "起始配置（股）": market.retail_initial_shares,
                             "目前持股（股）": sum(agent.shares for agent in market.agents),
                             "起始現金（元）": market.retail_initial_cash,
                             "目前現金（元）": round(sum(agent.cash for agent in market.agents), 2)}]
                for account in market.institutions.accounts:
                    balances.append({"交易者": account.sleeve, "起始配置（股）": account.initial_shares,
                                     "目前持股（股）": account.shares,
                                     "起始現金（元）": market.institutions.initial_cash[account.sleeve],
                                     "目前現金（元）": account.cash})
                for balance in balances:
                    balance["起始持股市值（元）"] = round(balance["起始配置（股）"] * market.stock.initial_price, 2)
                initial_total = sum(balance["起始配置（股）"] for balance in balances)
                st.caption(f"模擬起始流通股數 {initial_total:,} 股")
                if getattr(market, "initial_ownership", None) is not None:
                    foreign_pct, trust_pct = market.initial_ownership
                    st.caption(f"起始持股參考｜外資及陸資 {foreign_pct:.2f}%（2026-11-02 官方資料）｜投信 {trust_pct:.1f}%（模擬假設）")
                st.dataframe(balances, hide_index=True, alt="電腦交易者與法人起始配置及目前持股")
                st.caption("起始配置是模擬開局的股份，後續買賣會反映在目前持股。")
        held = player.positions.get(symbol)
        cash_slot = st.empty()
        shares_slot = st.empty()
        with st.expander("增加模擬資金"):
            with st.form("deposit_form"):
                amount = st.number_input("入金金額（元）", min_value=1, max_value=MAX_DEPOSIT,
                                         value=1_000_000, step=1, key="deposit_amount")
                deposit = st.form_submit_button("加入資金", key="deposit_cash", width="stretch")
            if deposit:
                player.deposit(amount, timestamp=world.timestamp)
                st.success(f"已加入 {amount:,} 元模擬資金。")
        if held and held.quantity:
            st.caption(f"持有 {held.quantity:,} 股｜均價 {held.cost / held.quantity:,.2f}")
        order_type = st.selectbox("委託方式", ["限價", "市價"], key="order_type")
        lower, upper = market.next_limits
        st.caption(f"{'下一交易日' if market.phase == 0 else '今日'}委託價格範圍：{lower:,.2f}–{upper:,.2f} 元")
        cents = round(market.price * 100)
        step = tick_cents(cents, bool(market.stock.basket))
        default_price = round(cents / step) * step / 100
        quantity = st.number_input("委託股數（1 股起）", min_value=1, max_value=1_000_000,
                                   value=10, step=1, key="order_quantity")
        buy_key, sell_key = f"buy_price_{symbol}", f"sell_price_{symbol}"
        st.session_state.setdefault(buy_key, default_price)
        st.session_state.setdefault(sell_key, default_price)
        limit_up, limit_down = st.columns(2)
        for column, label, price in ((limit_up, "漲停價", upper), (limit_down, "跌停價", lower)):
            if column.button(f"帶入{label} {price:,.2f}", key=f"{label}_{symbol}",
                             disabled=order_type == "市價", width="stretch"):
                st.session_state[buy_key] = price
                st.session_state[sell_key] = price
        buy_col, sell_col = st.columns(2)
        buy_price = buy_col.number_input("買進價格（元）", min_value=0.01, value=None,
                                        step=step / 100, format="%.2f", key=buy_key,
                                        disabled=order_type == "市價")
        sell_price = sell_col.number_input("賣出價格（元）", min_value=0.01, value=None,
                                          step=step / 100, format="%.2f", key=sell_key,
                                          disabled=order_type == "市價")
        if order_type == "市價":
            buy_qty, buy_gross = preview_sweep(asks, quantity)
            sell_qty, sell_gross = preview_sweep(bids, quantity)
        else:
            buy_qty = quantity if buy_price is not None else 0
            sell_qty = quantity if sell_price is not None else 0
            buy_gross, sell_gross = (buy_price or 0) * buy_qty, (sell_price or 0) * sell_qty
        buy_cost = fee(buy_gross)
        sell_fee = fee(sell_gross)
        sell_tax = sales_tax(sell_gross / sell_qty, sell_qty, bool(market.stock.basket)) if sell_qty else 0
        buy_col.caption(f"估 {buy_qty:,} 股支出 {buy_gross + buy_cost:,.2f} 元（手續費 {buy_cost:,.2f}）")
        sell_col.caption(f"估 {sell_qty:,} 股淨收 {sell_gross - sell_fee - sell_tax:,.2f} 元（費 {sell_fee:,.2f}／稅 {sell_tax:,.2f}）")
        buy = buy_col.button("買入", key="buy", type="primary", width="stretch",
                             disabled=order_type == "限價" and buy_price is None)
        sell = sell_col.button("賣出", key="sell", width="stretch",
                               disabled=order_type == "限價" and sell_price is None)
        if buy or sell:
            try:
                order = world.place_order(symbol, "買入" if buy else "賣出", int(quantity), order_type,
                                          (float(buy_price) if buy else float(sell_price)) if order_type == "限價" else None)
                price_text = f"，委託價 {order.limit_price:,.2f} 元" if order.limit_price is not None else ""
                st.success(f"委託 #{order.ident}：{order.order_type}{order.side} {order.quantity:,} 股{price_text}，等待符合價格成交。")
            except ValueError as error:
                st.error(str(error))
        recent_fill = next((trade for trade in reversed(player.trades) if trade["股票代碼"] == symbol), None)
        if recent_fill:
            st.caption(f"我的最近成交｜{recent_fill['成交時間']}｜{recent_fill['買賣']} "
                       f"{recent_fill['股數']:,} 股 × {recent_fill['成交價']:,.2f} 元"
                       f"＝ **{recent_fill['成交金額']:,.2f} 元**"
                       f"｜手續費 {recent_fill.get('手續費', 0):,.2f}｜證交稅 {recent_fill.get('證交稅', 0):,.2f}")
        st.caption(f"模擬手續費 {BROKER_FEE_RATE:.4%}。賣出估算先採一般稅率，實際依成交分價與當沖資格計算。"
                   "市價單在下一筆掃過可用檔位，餘量取消。限價單依價位和送單順序排隊，餘量收盤到期。")
        if quote.get("流動性狀態"):
            st.caption("本筆市場冷清，沒有成交。限價單會繼續排隊。")
        if quote.get("分析師跟單股數"):
            st.caption(f"分析師喊盤後，本筆模擬代理人跟單成交 {quote['分析師跟單股數']:,} 股。")
        if quote.get("玩家本筆淨成交股數"):
            player_flow = quote["玩家本筆淨成交股數"]
            player_impact = quote.get("玩家成交買賣力道貢獻 (百分點)", 0)
            st.caption(f"本筆玩家淨成交 {player_flow:+,} 股｜對參考價的推力 {player_impact:+.3f} 個百分點（限制與取整前）")
        if not playing:
            st.caption("行情暫停時，可按左側「下一筆報價」成交，或開啟自動跳動行情。")
        pending = player.pending
        if pending:
            cancel_id = st.selectbox(
                "取消待成交委託", [o.ident for o in pending],
                format_func=lambda ident: next(
                    f"#{o.ident} {stock_labels[o.symbol]} {o.order_type}{o.side} {o.quantity} 股" for o in pending if o.ident == ident),
                key="cancel_id",
            )
            if st.button("取消這筆委託", key="cancel_order", width="stretch"):
                world.cancel_order(cancel_id)
                st.success("委託已取消，預留的現金或股數已釋放。")
        cash_slot.markdown(f"可用現金 **{player.available_cash:,.2f} 元**")
        shares_slot.markdown(f"可賣股數 **{player.available_shares(symbol):,} 股**")
        if market.event:
            st.info(market.event.title, title="本股最新消息")
        st.caption(f"{market.stock.exchange}｜{market.stock.tag}。起始價採官方歷史收盤，後續 K 線由模擬產生。")
        if market.calibration:
            profile = market.calibration
            st.caption(f"校準樣本 {profile['first_date']} 至 {profile['last_date']}（{profile['sessions']} 日）"
                       f"｜日量中位數 {profile['median_volume']:,} 股｜日內高低振幅中位數 {profile['median_range']:.2%}"
                       f"｜日報酬波動 {profile['daily_volatility']:.2%}")

    st.subheader("我的資產")
    equity = player.wealth(world.markets)
    unrealized = sum(position.quantity * world.markets[code].price - position.cost
                     for code, position in player.positions.items())
    total_pl = equity - player.contributed_capital
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("總資產", f"{equity:,.2f}", f"{total_pl:+,.2f}", delta_color="inverse")
    a2.metric("可用現金", f"{player.available_cash:,.2f}")
    a3.metric("未實現損益", f"{unrealized:+,.2f}", f"{unrealized:+,.2f}", delta_color="inverse")
    a4.metric("已實現損益", f"{player.realized:+,.2f}", f"{player.realized:+,.2f}", delta_color="inverse")
    st.caption(f"初始資金 {player.initial_cash:,.0f} 元｜累計入金 {player.deposited_cash:,.0f} 元｜帳上現金 {player.cash:,.2f} 元｜買單預留 {player.cash - player.available_cash:,.2f} 元｜損益占投入資金 {total_pl / player.contributed_capital * 100:+.2f}%")
    st.caption(f"現金占總資產 {player.cash / equity * 100:.2f}%；下表持股占比以含現金的總資產計算。")
    daytrade = player.daytrade_summary(market.trading_date.isoformat())
    st.caption(f"本日已完成先買後賣：{daytrade['股數']:,} 股｜當沖配對淨損益：{daytrade['淨損益']:+,.2f} 元（依當日買單先進先出配對，扣除費稅）。")

    holdings = pd.DataFrame([
        {"代碼": world.markets[code].stock.code or code, "股票": world.markets[code].stock.name, "持股": position.quantity,
         "平均成本": position.cost / position.quantity, "最新價": world.markets[code].price,
         "市值": position.quantity * world.markets[code].price,
         "資產占比 (%)": position.quantity * world.markets[code].price / equity * 100,
         "未實現損益": position.quantity * world.markets[code].price - position.cost}
        for code, position in player.positions.items() if position.quantity
    ])
    if holdings.empty:
        st.info("目前沒有持股；選一支股票，輸入股數後按「買入」。")
    else:
        st.dataframe(styled_prices(holdings, ["未實現損益"]), hide_index=True, alt="我的股票持倉、成本、市值與損益")

    with st.expander("我的委託與成交紀錄", expanded=bool(player.pending)):
        if player.orders:
            st.dataframe(pd.DataFrame([
                {"編號": o.ident, "股票": stock_labels[o.symbol], "買賣": o.side, "股數": o.quantity,
                 "類型": o.order_type, "委託價": o.limit_price, "有效交易日": o.valid_day,
                 "狀態": o.status, "已成交股數": o.filled_quantity, "未成交股數": o.remaining,
                 "成交均價": o.fill_price, "最近成交時間": o.filled_at}
                for o in reversed(player.orders)
            ]), hide_index=True, alt="個人全部委託的狀態及成交價格")
        else:
            st.caption("尚無委託。")
        if player.trades:
            st.dataframe(pd.DataFrame(player.trades).iloc[::-1], hide_index=True, alt="個人逐筆成交與已實現損益")

    if mobile and not st.toggle("顯示全市場與研究明細", key="mobile_details",
                                help="開啟後可查看法人排行、全市場行情、研究工具與匯出。"):
        return

    st.subheader("全市場即時行情")
    board = pd.DataFrame([
        {"代碼": m.stock.code or code, "股票": m.stock.name, "市場": m.stock.exchange,
         "模擬風格": m.stock.tag, "波動係數": m.stock.volatility,
         "產業": m.stock.sector, "最新價": m.price,
         "指數占比 (%)": current_weights[code],
         "漲跌": m.price - m.history[-1]["昨收"], "漲跌幅 (%)": m.history[-1]["漲跌幅 (%)"],
         "開盤": m.history[-1]["開盤"], "最高": m.history[-1]["最高"], "最低": m.history[-1]["最低"],
         "成交量": m.history[-1]["成交量"],
         "狀態": "漲停" if m.price == m.history[-1]["漲停價"] else "跌停" if m.price == m.history[-1]["跌停價"] else m.history[-1]["狀態"]}
        for code, m in world.markets.items() if m.stock.active
    ])
    st.dataframe(styled_prices(board, ["漲跌", "漲跌幅 (%)"]), hide_index=True,
                 height=450, alt="台股模擬清單的代號、風格與即時報價")

    days = [row["交易日"] for row in reversed(world.index.history) if row["狀態"] != "起點"]
    with st.expander("每日漲跌幅與成交量排名", key="daily_rankings", on_change="rerun") as rankings:
        if rankings.open and days:
            ranking_day = st.selectbox("排名交易日", days, key="ranking_day")
            ranking_type = st.selectbox("排名方式", ["漲幅", "跌幅", "成交量"], key="ranking_type")
            rank_rows = []
            for m in world.markets.values():
                if not m.stock.active:
                    continue
                row = next((r for r in reversed(m.history)
                            if r["交易日"] == ranking_day and r["狀態"] != "起點"), None)
                if row is not None:
                    rank_rows.append({"代號": m.stock.code, "標的": m.stock.name, "狀態": row["狀態"],
                                      "漲跌幅 (%)": row["漲跌幅 (%)"], "成交量（股）": row["成交量"]})
            column = "成交量（股）" if ranking_type == "成交量" else "漲跌幅 (%)"
            ranking = pd.DataFrame(rank_rows).sort_values([column, "代號"], ascending=[ranking_type == "跌幅", True])
            ranking.insert(0, "排名", range(1, len(ranking) + 1))
            st.caption("漲跌幅以昨收為基準。盤中數字為目前累計，每筆成交只計一次股數。")
            st.dataframe(styled_prices(ranking, ["漲跌幅 (%)"]), hide_index=True, height=420,
                         alt="指定交易日的漲幅、跌幅或成交量排名")

    with st.expander("逐筆成交明細（含買方與賣方）", key="trade_ledger", on_change="rerun") as ledger:
        if ledger.open and days:
            trade_day = st.selectbox("成交交易日", days, key="market_trade_day")
            trades = pd.DataFrame(market.trade_records(trade_day))
            st.caption(f"{stock_labels[symbol]}｜做市商承接模擬委託，成交的買方與賣方股數相同。")
            if trades.empty:
                st.caption("這天沒有可顯示的逐筆成交。升級前的舊行情保留原有日成交量。")
            else:
                volume = trades["成交股數"].sum()
                st.caption(f"明細共 {len(trades):,} 筆｜買方 {volume:,} 股｜賣方 {volume:,} 股｜成交量 {volume:,} 股")
                daily_volume = sum(r["成交量"] for r in market.history if r["交易日"] == trade_day)
                if volume < daily_volume:
                    st.caption(f"這天有 {daily_volume - volume:,} 股成交發生在升級前，只有日成交量。")
                st.dataframe(trades, hide_index=True, height=400, alt="逐筆成交的買方、賣方、股數與價格")
                st.download_button("下載此標的當日成交 CSV", trades.to_csv(index=False).encode("utf-8-sig"),
                                   f"afs_{market.stock.code}_{trade_day}_fills.csv", "text/csv")

    with st.expander("三大法人籌碼與排行（模擬）", expanded=True):
        st.caption("外資＋投信＋自營商＝三大法人；自營商為自行買賣＋避險帳戶合計，不重複計算。盤中為累計，連買／連賣及5／20日統計只採已收盤資料。")
        st.markdown("[分類參考：證交所三大法人買賣超日報](https://www.twse.com.tw/zh/trading/foreign/t86.html)；本頁數值均由模擬成交產生。")
        st.dataframe(styled_prices(pd.DataFrame(institution_rows), ["買賣超股數", "買賣超金額", "近5收盤日買賣超", "近20收盤日買賣超"]),
                     hide_index=True, alt="所選股票三大法人買賣超、持股及連續買賣日數")
        st.caption("持股占比分母為這支股票的模擬代理人、法人及玩家持股總和，不含虛擬做市商，也不是占真實公司發行股數的比例。連買日數正值、連賣負值；無舊資料不補算。")
        plans = [{key: row.get(key) for key in ("法人", "策略", "計畫目標股數", "計畫剩餘股數", "買賣超股數")}
                 for row in market.institutions.current.values() if "計畫目標股數" in row]
        if plans:
            st.dataframe(plans, hide_index=True, alt="外資投信跨日拆單計畫和執行進度")
        dealer = [dict(row) for row in market.institutions.current.values() if row["法人"] == "自營商"]
        if dealer:
            st.dataframe(pd.DataFrame(dealer), hide_index=True,
                         alt="自營商自行買賣與避險庫存代理的分項成交")
        st.caption("避險分類使用庫存調整代理策略，沒有模擬權證、選擇權或真正的衍生品避險。各股法人資金獨立，尚非跨股票基金配置。")
        selected_group = st.selectbox("法人走勢類別", ["三大法人合計", "外資", "投信", "自營商"], key="institution_group")
        grouped = pd.DataFrame(market.institutions.grouped_history())
        if not grouped.empty:
            flow = grouped.groupby("交易日")["買賣超股數"].sum() if selected_group == "三大法人合計" else grouped[grouped["法人"] == selected_group].set_index("交易日")["買賣超股數"]
            st.line_chart(flow.tail(60), height=200, alt="最近六十個交易日的模擬法人每日買賣超")
        ranking = []
        for code, current_market in world.markets.items():
            if not current_market.stock.active:
                continue
            owned = player.positions[code].quantity if code in player.positions else 0
            rows = current_market.institution_snapshot(owned)
            ranking.append({"代碼": current_market.stock.code, "股票": current_market.stock.name,
                            **{row["法人"]: row["買賣超股數"] for row in rows},
                            "三大法人合計": sum(row["買賣超股數"] for row in rows)})
        sort_column = st.selectbox("排行依據", ["三大法人合計", "外資", "投信", "自營商"], key="institution_ranking")
        ascending = st.toggle("改看賣超排行", value=False, key="institution_sellers")
        st.dataframe(styled_prices(pd.DataFrame(ranking).sort_values(sort_column, ascending=ascending),
                                   ["三大法人合計", "外資", "投信", "自營商"]), height=300, hide_index=True,
                     alt="全市場模擬外資、投信、自營商買超或賣超排行")

    with st.expander("逐筆漲跌紀錄與新聞時間線"):
        st.caption("本筆漲跌相對上一筆成交；主畫面的今日漲跌仍相對昨收。分解是模型內衝擊的換算記帳，不是獨立新聞因果估計。")
        if market.last_explanation:
            st.dataframe(pd.DataFrame([{"來源": key, "報酬貢獻（百分點）": value}
                                       for key, value in market.last_explanation["分解"].items()]), hide_index=True,
                         alt="本筆主動需求、共同波動、個股擾動與價格限制的報酬分解")
        st.dataframe(pd.DataFrame(market.quotes[-60:]).reindex(columns=[
            "交易日", "時間", "前價", "股價", "本筆漲跌 (%)", "主動淨需求", "成交量", "漲跌說明", "作用新聞"]),
            hide_index=True, height=260, alt="所選股票最近六十筆漲跌與模擬原因")
        if world.news_feed:
            st.dataframe(pd.DataFrame(world.news_feed[-30:][::-1]).reindex(columns=[
                "發布時間", "來源", "影響範圍", "方向", "強度", "目標價（元）", "發布時參考價（元）",
                "利率變動 (bp)", "模擬利率 (%)", "標的數", "消息"]),
                hide_index=True, height=260, alt="全市場手動及隨機模擬新聞時間線")
        else:
            st.caption("目前沒有市場新聞，可在左側立即產生一則。")

    with st.expander("加權指數成分占比與日 K"):
        render_index_composition(composition)
        render_candles(index.history, "index", world.timestamp, fixed=review_chart,
                       price_title="指數點數", alt="AFS 加權指數的可拖曳日 K 與全市場成交量")

    with st.expander("多股走勢比較"):
        compared = st.multiselect("選擇比較股票", list(stock_labels), default=list(stock_labels)[:4],
                                  format_func=stock_labels.get, key="compared_stocks")
        if compared:
            normalized = pd.DataFrame({
                stock_labels[code]: pd.Series({
                    row["交易日"]: row["股價"] / world.markets[code].stock.initial_price * 100
                    for row in world.markets[code].history
                })
                for code in compared
            }).sort_index()
            normalized.index.name = "交易日"
            st.line_chart(normalized, height=230, alt="各股以起點一百換算的相對表現")

    with st.expander("AI 投資者觀察室"):
        st.caption("各股有獨立的離線數值交易者；決策先產生預測，成交後才更新誤差，沒有使用未來行情。")
        st.dataframe(pd.DataFrame([
            {"ID": a.ident, "性格": a.style, "決策": a.action, "股數": abs(a.last_qty),
             "原因": a.reason, "模型": a.model_family, "人物": a.personality,
             "信念 (%)": round(a.belief * 100, 3), "下一筆預測價": round(a.forecast, 2) if a.model_family not in ("ZI-C 簡化", "原始規則基準") else None,
             "成本錨點": round(a.cost_basis, 2), "預測誤差": a.forecast_error,
             "恐懼": round(a.fear, 2), "持股": a.shares, "資產": round(a.wealth(market.price), 2)}
            for a in market.agents
        ]), hide_index=True, height=260, alt="本股 AI 交易者的最新決策與資產")
        if market.model_config.mode != "legacy" and market.research_engine:
            engine = market.research_engine
            st.caption(f"Santa Fe 規則演化：{engine.generations} 代｜非 ZI 交易者最新單筆預測 MSE：{engine.forecast_mse if engine.forecast_mse is not None else '無'}")
            ident = st.selectbox("查看一名交易者的分析紀錄", list(range(market.n)),
                                format_func=lambda i: f"#{i + 1:03d} · {market.agents[i].personality}", key="research_agent")
            if engine.zi_mask[ident]:
                st.caption("這位是零智慧交易者，隨機出限價單，不採用基本面、消息或預測訊號。")
            elif engine.last_reports:
                st.dataframe(pd.DataFrame([{"角色": role, "訊號／上限 (%)": float(values[ident]) * 100}
                                           for role, values in engine.last_reports.items()]), hide_index=True,
                             alt="所選交易者的基本面、技術、消息、多空與風控數值報告")

    with st.expander("A/B 新聞實驗"):
        st.caption("相同種子與外部衝擊，從獨立起點運作 30 個交易日；對照市場關閉隨機新聞，只比較控制台指定的消息。")
        if st.button("執行 A/B 實驗", key="run_ab"):
            a_world = Exchange(seed=73, volatility=world.volatility, model_config=world.model_config)
            b_world = Exchange(seed=73, volatility=world.volatility, model_config=world.model_config)
            a_world.announce(headline or "模擬新聞", tone, power, reach, truth, anchor, target)
            a_world.advance(30)
            b_world.advance(30)
            st.session_state.comparison = {
                "description": f"{stock_labels[symbol]}｜新聞範圍：{stock_labels.get(target, '全市場')}｜波動 {world.volatility:.1f} 倍",
                "prices": pd.DataFrame({
                    "交易日": [row["交易日"] for row in a_world.markets[symbol].history],
                    "A：有新聞": [row["股價"] for row in a_world.markets[symbol].history],
                    "B：無新聞": [row["股價"] for row in b_world.markets[symbol].history],
                }).set_index("交易日"),
            }
        if st.session_state.comparison is not None:
            st.caption(st.session_state.comparison["description"])
            st.line_chart(st.session_state.comparison["prices"], alt="三十個交易日的新聞對照實驗")

    with st.expander("論文模型對照與消融實驗"):
        st.caption("在獨立單股市場比較原始規則、零智慧、完整混合及五組移除機制版本；同種子、同新聞與個股擾動，不含全市場／產業共同衝擊，不改動你的帳戶。")
        measured_days = st.number_input("對照交易日數（另加20日暖身）", 20, 250, 60, 10, key="experiment_days")
        seed_count = st.number_input("對照亂數種子數", 2, 20, 4, 1, key="experiment_seeds")
        if market.stock.basket:
            st.caption("單股模型對照請選擇個股。ETF 的籃子追蹤可用上方 A/B 實驗觀察。")
        if st.button("執行模型對照", key="run_models", disabled=bool(market.stock.basket)):
            active = world.model_config
            config = ModelConfig.preset("hybrid", social_strength=active.social_strength,
                                        loss_aversion=active.loss_aversion, exogenous_scale=active.exogenous_scale,
                                        evolve_every=active.evolve_every, error_alpha=active.error_alpha,
                                        prospect_fraction=active.prospect_fraction)
            signature = hashlib.sha256(b"".join((Path(__file__).resolve().parent / name).read_bytes()
                                                for name in ("market.py", "research_models.py", "model_experiments.py", "market_news.py", "institutions.py"))).hexdigest()
            with st.spinner("計算配對種子與消融版本…"):
                st.session_state.research_comparison = cached_model_comparison(
                    symbol, tuple(7 + 37 * i for i in range(int(seed_count))), int(measured_days),
                    json.dumps(config.to_dict(), sort_keys=True), signature)
        if st.session_state.research_comparison:
            result = st.session_state.research_comparison
            st.caption(f"觀察標的：{result['metadata']['symbol']}｜{result['metadata']['days']} 日 × {len(result['metadata']['seeds'])} 種子｜每股32名交易者｜外部亂數一致性已檢查")
            st.dataframe(pd.DataFrame(result["summary"]).rename(columns={
                "variant": "版本", "daily_vol_pct": "日波動 (%)", "excess_kurtosis": "超額峰度",
                "abs_return_acf1": "絕對報酬自相關", "max_drawdown_pct": "最大回撤 (%)",
                "mean_daily_volume": "日均成交股數", "forecast_mse": "單筆預測MSE",
                "limit_hit_pct": "觸及漲跌停 (%)"}), hide_index=True, alt="多種子模型對照的市場統計指標")
            st.dataframe(pd.DataFrame(result["paired"]), hide_index=True, height=220,
                         alt="相對原始規則基準的配對差異與百分之九十五種子自助抽樣區間")
            st.caption("這些是模擬內診斷。區間為配對種子 bootstrap，非真實台股驗證；不能由較大波動或較低回撤直接推論模型更準。原始規則與純 ZI 沒有單筆預測 MSE。")
            st.download_button("下載模型對照 CSV", pd.DataFrame(result["runs"]).to_csv(index=False).encode("utf-8-sig"),
                               "afs_model_comparison.csv", "text/csv", key="download_models")
            st.download_button("下載實驗設定 JSON", json.dumps(result["metadata"], ensure_ascii=False, indent=2),
                               "afs_model_config.json", "application/json", key="download_model_config")
        references = json.loads((Path(__file__).resolve().parent / "research" / "references.json").read_text(encoding="utf-8"))
        for reference in references:
            st.markdown(f"[{reference['title']}]({reference['url']}) — {reference['authors']}，{reference['year']}")
        st.caption("Santa Fe 與 ZI 使用目前做市商市場的簡化適配；沒有重現原文的均衡撮合或雙向拍賣。StockAgent／TradingAgents 未啟用 LLM，也未複製原文報酬結果。")

    with st.expander("匯出研究與交易紀錄"):
        if st.button("準備本次匯出", key="prepare_export"):
            sources = [
                ("下載我的委託 CSV", "afs_my_orders.csv", lambda: orders_csv(player.orders, world.markets)),
                ("下載我的成交 CSV", "afs_my_trades.csv", lambda: pd.DataFrame(player.trades)),
                ("下載日 K CSV", "afs_daily_prices.csv", lambda: pd.DataFrame(world.records("history"))),
                ("下載加權指數 CSV", "afs_weighted_index.csv", lambda: pd.DataFrame(world.index.history)),
                ("下載指數占比 CSV", "afs_index_weights.csv", lambda: composition),
                ("下載 AI 交易 CSV", "afs_trades.csv", lambda: logs_csv(world)),
                ("下載新聞 CSV", "afs_events.csv", lambda: pd.DataFrame(world.records("events"))),
                ("下載逐筆漲跌原因 CSV", "afs_quote_reasons.csv", lambda: pd.DataFrame(world.records("quotes"))),
                ("下載市場新聞時間線 CSV", "afs_news_timeline.csv", lambda: pd.DataFrame(world.news_feed)),
                ("下載三大法人日報 CSV", "afs_institutions.csv", lambda: pd.DataFrame(world.institution_records())),
                ("下載法人分項 CSV", "afs_institution_details.csv", lambda: pd.DataFrame(world.institution_records(detail=True))),
            ]
            files = []
            for label, filename, build in sources:
                result = build()
                data = result if isinstance(result, bytes) else result.to_csv(index=False).encode("utf-8-sig")
                files.append((label, filename, data))
                del result
            st.session_state.exports = {"timestamp": world.timestamp, "files": files}
        if st.session_state.exports:
            export = st.session_state.exports
            st.caption(f"匯出快照：{export['timestamp']}。請在再次準備前下載這次的委託、成交與行情。")
            for label, filename, data in export["files"]:
                st.download_button(label, data, filename, "text/csv", on_click="ignore")


@st.fragment
def historical_records():
    submitted_tables = {}
    with st.expander("匯入 CSV／查看歷史場次"):
        st.caption("同一場次的檔案可分次上傳。要換場次，請先按下方的「清除歷史檢視」。")
        with st.form("historical_csv", clear_on_submit=True):
            uploads = {
                "委託": st.file_uploader("委託 CSV", type="csv", key="historical_orders"),
                "成交": st.file_uploader("成交 CSV", type="csv", key="historical_trades"),
                "行情": st.file_uploader("行情 CSV（日 K 或逐筆報價）", type="csv", key="historical_prices"),
            }
            submitted = st.form_submit_button("匯入並查看")

        if submitted:
            problems = []
            for kind, upload in uploads.items():
                if upload is not None:
                    try:
                        submitted_tables[kind] = (upload.name, read_history_csv(upload.getvalue(), kind))
                    except ValueError as error:
                        problems.append(f"{kind}：{error}")
            if problems:
                for problem in problems:
                    st.error(problem)
            elif not submitted_tables:
                st.warning("請先選擇至少一份 CSV。")
            else:
                st.session_state.historical_tables = {
                    **st.session_state.get("historical_tables", {}), **submitted_tables,
                }
                for kind in submitted_tables:
                    st.session_state.pop(f"historical_page_{kind}", None)
                if "行情" in submitted_tables:
                    st.session_state.pop("historical_symbol", None)
                st.success("歷史資料已載入。")

    tables = st.session_state.get("historical_tables")
    if "exchange_before_history" in st.session_state:
        if st.button("切回接續前的交易台", key="restore_history_exchange"):
            st.session_state.history_exchange_action = "restore"
            st.rerun()
    if not tables:
        return
    st.header("已匯入的歷史場次")
    if st.button("清除歷史檢視", key="clear_history_csv"):
        st.session_state.pop("historical_tables", None)
        return

    for kind in ("委託", "成交"):
        st.subheader(f"歷史{kind}紀錄")
        if kind in tables:
            render_historical_table(kind, *tables[kind])
        else:
            st.caption(f"尚未上傳{kind} CSV。")

    st.subheader("歷史 K 線")
    if "行情" not in tables:
        st.caption("上傳日 K 或逐筆報價 CSV 後即可查看。")
        return
    name, prices = tables["行情"]
    st.caption("接續後，主盤會保留匯入的 K 線，並從最後一日的價格與下一個交易日繼續。"
               "若最後一天只到盤中，採用檔案中的最後價格。舊委託與成交留在這裡查閱，"
               "接續交易使用新的模擬帳戶，加權指數以匯入的個股行情估算。")
    if st.button("用這份行情接續主盤", key="resume_history_exchange"):
        st.session_state.history_exchange_action = "resume"
        st.rerun()
    label_columns = [column for column in ("台股代號", "股票名稱") if column in prices]
    instruments = prices[["股票代碼", *label_columns]].drop_duplicates("股票代碼")
    labels = {row[0]: " · ".join(part for part in row[1:] if part) or row[0]
              for row in instruments.itertuples(index=False, name=None) if row[0]}
    if not labels:
        st.info("行情 CSV 沒有可查看的標的。")
        return
    symbol = st.selectbox("歷史行情標的", list(labels), format_func=labels.get, key="historical_symbol")
    selected = prices.loc[prices["股票代碼"].eq(symbol)]
    bars = historical_daily_prices(selected)
    if bars.empty:
        st.info("這個標的沒有可繪圖的交易日價格。")
    elif set(OHLC).issubset(bars):
        bars = bars.assign(狀態="歷史")
        if "成交量" not in bars:
            bars["成交量"] = None
        rows = bars.astype(object).where(bars.notna(), None).to_dict("records")
        st.caption(f"{bars['交易日'].iloc[0]} 至 {bars['交易日'].iloc[-1]} · {len(bars):,} 個交易日")
        st.echarts_chart(candle_chart(rows), height=520, key=f"historical_candles_{symbol}",
                         alt=f"{labels[symbol]} 的歷史日 K 與成交量")
    else:
        st.caption("這份行情只有收盤價，以下顯示歷史收盤走勢。")
        st.line_chart(bars, x="交易日", y="收盤", alt=f"{labels[symbol]} 的歷史收盤價")

    with st.expander("查看原始行情紀錄", key="historical_raw_prices", on_change="rerun") as details:
        if details.open:
            render_historical_table("行情", name, selected)


def render_historical_table(kind, name, frame):
    st.caption(f"{name}｜{len(frame):,} 筆")
    if frame.empty:
        st.info("這份檔案沒有資料列。")
        return
    page_size = 100
    page_count = (len(frame) + page_size - 1) // page_size
    page = 1
    if page_count > 1:
        page = st.number_input(f"歷史{kind}頁碼（共 {page_count:,} 頁）", min_value=1, max_value=page_count,
                               key=f"historical_page_{kind}")
    start = (page - 1) * page_size
    st.dataframe(frame.iloc[start:start + page_size], hide_index=True, alt=f"匯入的歷史{kind}資料")


trading_desk()
historical_records()
st.caption("模擬交易遊戲；價格、流動性與成交規則為模擬設定，沒有連接券商或真實行情。")

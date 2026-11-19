"""Native Vega charts for simulated daily candles and intraday prints."""
import altair as alt
import pandas as pd

from market import TICKS_PER_DAY

UP = "#ef5350"
DOWN = "#26a69a"
FLAT = "#9e9e9e"


def index_technicals(history):
    closed = [row for row in history if row["狀態"] == "收盤"]
    if not closed:
        return pd.DataFrame()
    frame = pd.DataFrame(closed).set_index("交易日")[["收盤"]].astype(float)
    close = frame["收盤"]
    for window in (5, 20, 60):
        frame[f"MA{window}"] = close.rolling(window).mean()
    delta = close.diff()
    # Wilder smoothing starts at the simple mean of the first 14 changes.
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    if len(close) >= 15:
        for values in (gain, loss):
            values.iloc[14] = values.iloc[1:15].mean()
            values.iloc[:14] = float("nan")
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        total = avg_gain + avg_loss
        frame["RSI14"] = (100 * avg_gain / total).mask(total == 0, 50)
    else:
        frame["RSI14"] = float("nan")
    frame["DIF"] = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(
        span=26, adjust=False, min_periods=26).mean()
    frame["訊號線"] = frame["DIF"].ewm(span=9, adjust=False, min_periods=9).mean()
    frame["MACD 柱"] = frame["DIF"] - frame["訊號線"]
    return frame


def candle_chart(history, lookback=60, price_title="價格（元）"):
    """A bounded, pannable financial chart; all sessions remain available."""
    rows = [row for row in history if row["狀態"] != "起點"]
    if not rows:
        return None
    closes = pd.Series([row["收盤"] for row in rows])
    def moving_average(window):
        return [None if pd.isna(value) else round(value, 4)
                for value in closes.rolling(window).mean()]
    dates = [row["交易日"] for row in rows]
    axes = [{"type": "category", "data": dates, "gridIndex": i,
             "boundaryGap": True, "axisLabel": {"show": i == 1, "hideOverlap": True, "showMaxLabel": True},
             "axisPointer": {"show": True}} for i in range(2)]
    zoom = {"xAxisIndex": [0, 1], "filterMode": "filter",
            "startValue": max(0, len(rows) - lookback), "endValue": len(rows) - 1,
            "minValueSpan": min(3, len(rows) - 1)}
    return {
        "animation": False,
        "legend": {"top": 0, "right": 5, "data": ["日 K", "MA5", "MA20", "成交量"]},
        "toolbox": {"left": 0, "top": 0, "feature": {"restore": {"title": "重設視角"}}},
        "tooltip": {"trigger": "axis", "confine": True, "axisPointer": {"type": "cross"}},
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "grid": [
            {"left": 12, "right": 22, "top": 55, "height": 290, "containLabel": True},
            {"left": 12, "right": 22, "top": 365, "height": 105, "containLabel": True},
        ],
        "xAxis": axes,
        "yAxis": [
            {"type": "value", "gridIndex": 0, "name": price_title, "scale": True,
             "boundaryGap": ["8%", "8%"], "axisLabel": {"hideOverlap": True}},
            {"type": "value", "gridIndex": 1, "name": "成交量（股）", "scale": True,
             "splitNumber": 2, "axisLabel": {"hideOverlap": True}},
        ],
        "dataZoom": [
            {**zoom, "type": "inside", "zoomOnMouseWheel": True, "moveOnMouseMove": True,
             "moveOnMouseWheel": False},
            {**zoom, "type": "slider", "bottom": 12, "height": 25, "showDetail": False},
        ],
        "series": [
            {"name": "日 K", "type": "candlestick", "clip": True, "cursor": "grab",
             "dimensions": ["開盤", "收盤", "最低", "最高"], "barMaxWidth": 18,
             "data": [[r["開盤"], r["收盤"], r["最低"], r["最高"]] for r in rows],
             "itemStyle": {"color": UP, "color0": DOWN, "borderColor": UP,
                           "borderColor0": DOWN, "borderColorDoji": FLAT}},
            *[{"name": name, "type": "line", "data": moving_average(window),
               "showSymbol": False, "clip": True, "lineStyle": {"width": 1.4, "color": color},
               "itemStyle": {"color": color}}
              for name, window, color in (("MA5", 5, "#ffb74d"), ("MA20", 20, "#64b5f6"))],
            {"name": "成交量", "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1,
             "clip": True, "barMaxWidth": 18,
             "data": [{"value": r["成交量"], "itemStyle": {
                 "color": UP if r["收盤"] > r["開盤"] else DOWN if r["收盤"] < r["開盤"] else FLAT}}
                 for r in rows]},
        ],
    }


def intraday_chart(quotes, previous_close, *, price_title="價格（元）"):
    if not quotes:
        return None
    latest_date = quotes[-1]["交易日"]
    frame = pd.DataFrame([row for row in quotes[-TICKS_PER_DAY:] if row["交易日"] == latest_date])
    color = UP if frame.iloc[-1]["股價"] > previous_close else DOWN if frame.iloc[-1]["股價"] < previous_close else FLAT
    line = alt.Chart(frame).mark_line(color=color, point=True).encode(
        x=alt.X("時間:O", title="盤中時間", axis=alt.Axis(labelAngle=0, labelOverlap="greedy")),
        y=alt.Y("股價:Q", title=price_title, scale=alt.Scale(zero=False)),
        tooltip=["交易日:N", "時間:O", alt.Tooltip("股價:Q", format=",.2f"), "成交量:Q"],
    )
    reference = alt.Chart(pd.DataFrame({"昨收": [previous_close]})).mark_rule(
        color=FLAT, strokeDash=[5, 5]).encode(y="昨收:Q")
    return (line + reference).properties(height=180)

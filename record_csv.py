"""Read-only CSV exports and historical records."""

import csv
from io import BytesIO, StringIO, TextIOWrapper

import pandas as pd


ORDER_COLUMNS = (
    "委託編號", "股票代碼", "台股代號", "股票名稱", "買賣", "原始股數", "委託類型",
    "委託價", "下單時間", "有效交易日", "狀態",
)
OHLC = ("開盤", "最高", "最低", "收盤")
HISTORY_COLUMNS = {
    "委託": ({"委託編號", "股票代碼", "買賣", "委託類型", "有效交易日", "狀態"},),
    "成交": ({"委託編號", "股票代碼", "買賣", "成交時間", "股數", "成交價"},),
    "行情": ({"股票代碼", "交易日", "收盤"}, {"股票代碼", "交易日", "時間", "股價"}),
}


def orders_csv(orders, markets):
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(ORDER_COLUMNS)
    for order in orders:
        stock = markets[order.symbol].stock
        writer.writerow((
            order.ident, order.symbol, stock.code, stock.name, order.side, order.quantity,
            order.order_type, order.limit_price, order.submitted_at, order.valid_day, order.status,
        ))
    return output.getvalue().encode("utf-8-sig")


def read_history_csv(data, kind):
    for encoding in ("utf-8-sig", "cp950"):
        try:
            with TextIOWrapper(BytesIO(data), encoding=encoding, newline="") as source:
                return _read_table(source, kind)
        except UnicodeDecodeError:
            continue
        except csv.Error as error:
            raise ValueError("CSV 格式有誤，請檢查引號、分隔符號或重新匯出。") from error
    raise ValueError("無法讀取檔案編碼，請使用 UTF-8 或繁體中文 Big5 CSV。")


def historical_daily_prices(quotes):
    """Normalize daily bars or build them from the available intraday quotes."""
    columns = ["交易日", *(["成交量"] if "成交量" in quotes else [])]
    if {"時間", "股價"}.issubset(quotes):
        prices = quotes[[*columns, "時間", "股價"]].copy()
        timestamps = pd.to_datetime(prices["交易日"] + " " + prices["時間"], format="mixed", errors="coerce")
        prices["時間"] = timestamps
        prices["股價"] = pd.to_numeric(prices["股價"], errors="coerce")
        prices = prices.loc[prices["股價"].between(0, float("inf"), inclusive="neither")].dropna(subset=["時間"])
        prices["交易日"] = prices["時間"].dt.strftime("%Y-%m-%d")
        prices = prices.sort_values("時間", kind="stable")
        daily = prices.groupby("交易日")
        bars = daily["股價"].ohlc().rename(columns=dict(zip(("open", "high", "low", "close"), OHLC)))
        if "成交量" in prices:
            bars["成交量"] = pd.to_numeric(prices["成交量"], errors="coerce").groupby(prices["交易日"]).sum(min_count=1)
        return bars.reset_index()

    days = quotes if "狀態" not in quotes else quotes.loc[quotes["狀態"].ne("起點")]
    if "昨收" in days:
        columns.append("昨收")
    price_columns = list(OHLC) if set(OHLC).issubset(days) else ["收盤"]
    bars = days[[*columns, *price_columns]].copy()
    bars["交易日"] = pd.to_datetime(bars["交易日"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    for column in bars.columns.drop("交易日"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    valid = bars[price_columns].gt(0) & bars[price_columns].lt(float("inf"))
    return (bars.loc[valid.all(axis=1)].dropna(subset=["交易日"])
            .drop_duplicates("交易日", keep="last").sort_values("交易日"))


def resume_price_history(prices, markets):
    """Convert the imported file to daily records keyed by the current catalogue."""
    symbols_by_code = {market.stock.code: symbol for symbol, market in markets.items() if market.stock.code}
    history = {}
    for symbol, frame in prices.groupby("股票代碼", sort=False):
        codes = frame["台股代號"] if "台股代號" in frame else ()
        stock_code = next((code for code in codes if code), "")
        current_symbol = symbols_by_code.get(stock_code or symbol)
        if current_symbol is None and not stock_code and symbol in markets:
            current_symbol = symbol
        if current_symbol is None:
            continue
        bars = historical_daily_prices(frame)
        if bars.empty:
            continue
        if not set(OHLC).issubset(bars):
            raise ValueError("接續主盤需要完整日 K 或逐筆報價。只有收盤價的檔案可用於下方歷史檢視。")
        if "成交量" not in bars or not bars["成交量"].between(0, float("inf"), inclusive="left").all():
            raise ValueError("接續主盤需要有效的成交量，請使用 AFS 原始匯出的行情 CSV。")
        history[current_symbol] = bars.to_dict("records")
    if not history:
        raise ValueError("行情檔裡找不到可接續的 AFS 標的與交易日。")
    return history


def _read_table(source, kind):
    reader = csv.reader(source, strict=True)
    columns = [column.strip() for column in next(reader, ())]
    if not columns or not all(columns) or len(set(columns)) != len(columns):
        raise ValueError("找不到有效欄名，請選擇 AFS 匯出的 CSV。")
    if not any(required.issubset(columns) for required in HISTORY_COLUMNS[kind]):
        raise ValueError(f"欄位與 AFS 的{kind} CSV 不符，請確認檔案放在正確的欄位。")

    def records():
        for row in reader:
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError(f"第 {reader.line_num} 行的欄位數與標題不符。")
            yield row

    # Keep values as text so ETF codes such as 0050 retain leading zeros.
    return pd.DataFrame.from_records(records(), columns=columns)

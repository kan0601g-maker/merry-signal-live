# web/scripts/wave3_live_update.py
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# CONFIG（神コードLIVE仕様）
# =========================
TICKERS = ["^GSPC", "GLD", "^N225", "VT", "SLV",
           "XLP", "XLV", "XLU", "IBB", "SMH", "QQQ", "ITA"]

CASH_TICKER = "SHY"

DATA_START = "1985-01-01"
BT_START = "2000-01-01"

FEE_RATE = 0.0002
SLIPPAGE = 0.0

W_LOOKBACK = 260
MIN_TOUCHES = 3
TOUCH_TOL = 0.005
RETEST_TOL = 0.003
TOUCH_WINDOW = 8
MIN_TOUCHES_IN_WINDOW = 2
LOWER_BODY_BREAK_TOL = 0.0

M_LOOKBACK = 12
MA200_PERIOD = 120

LEV = 3.0  # ←固定

PRIORITY_MODE = "RISK_ADJ"
PRIORITY_LIST = ["^GSPC", "VT", "GLD", "^N225", "SLV"]

ATR_PERIOD = 14

# =========================
# OUTPUT PATH
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "public", "data")

STATE_JSON = os.path.join(OUT_DIR, "state.json")
SIGNAL_JSON = os.path.join(OUT_DIR, "trade_signal.json")
SNAP_JSON = os.path.join(OUT_DIR, "universe_snapshot.json")
ACTIONS_CSV = os.path.join(OUT_DIR, "actions.csv")


# =========================
# UTILS
# =========================
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_state():
    if not os.path.exists(STATE_JSON):
        return {
            "mode": "LIVE",
            "asof": None,
            "lev": LEV,
            "current": None,
            "entry_date": None,
            "entry_price": None,
            "stop_price": None,
            "realized_equity": 1.0,
            "realized_pnl": 0.0
        }

    # BOM対応
    with open(STATE_JSON, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def append_action(row):
    header = not os.path.exists(ACTIONS_CSV)
    pd.DataFrame([row]).to_csv(
        ACTIONS_CSV,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8"
    )


# =========================
# DATA
# =========================
def download_daily(ticker):
    df = yf.download(ticker, start=DATA_START,
                     interval="1d", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]]


def to_weekly(df):
    return df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()


def to_monthly(df):
    return df.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()


# =========================
# 神コードロジック
# =========================
def find_horizontal_wick_zone(high_arr):
    n = len(high_arr)
    if n < MIN_TOUCHES:
        return np.nan

    start = max(0, n - W_LOOKBACK)
    seg = high_arr[start:n]

    for i in range(len(seg) - 1, -1, -1):
        level = seg[i]
        rel = np.abs(seg - level) / level
        touches = np.where(rel <= TOUCH_TOL)[0]
        if len(touches) >= MIN_TOUCHES:
            idx = touches + start
            return float(np.max(high_arr[idx]))

    return np.nan


def build_wave3_monthly_gate(df):
    m = to_monthly(df)
    m["MA200"] = m["Close"].rolling(MA200_PERIOD).mean()

    gate = np.zeros(len(m), dtype=bool)
    locked = False
    res = np.nan

    lows = m["Low"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    closes = m["Close"].to_numpy(float)
    ma200 = m["MA200"].to_numpy(float)

    for i in range(len(m)):
        if i < M_LOOKBACK or np.isnan(ma200[i]):
            continue

        s = i - M_LOOKBACK
        prev_low = np.min(lows[s:i])
        window_high = np.max(highs[s:i+1])

        if not locked and lows[i] < prev_low:
            locked = True
            res = window_high
            continue

        if locked:
            if closes[i] > res and closes[i] > ma200[i]:
                locked = False
                res = np.nan
                gate[i] = True
        else:
            gate[i] = closes[i] > ma200[i]

    return pd.Series(gate, index=m.index)


def weekly_atr_pct(df):
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()

    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan

    tr = np.maximum(h - l,
                    np.maximum(np.abs(h - prev_c),
                               np.abs(l - prev_c)))

    atr = pd.Series(tr, index=df.index).rolling(
        ATR_PERIOD).mean().to_numpy()

    return atr / c


# =========================
# MAIN
# =========================
def main():

    ensure_dirs()
    state = load_state()
    state["lev"] = LEV

    signals = {}
    for t in TICKERS:
        df = download_daily(t)
        w = to_weekly(df)
        w = w[w.index >= BT_START]

        zone = np.full(len(w), np.nan)
        h = w["High"].to_numpy()

        for i in range(len(w)):
            zone[i] = find_horizontal_wick_zone(h[:i+1])

        w["Zone"] = zone
        w["ATR_PCT"] = weekly_atr_pct(w)

        m_gate = build_wave3_monthly_gate(df)
        w["WaveOK"] = m_gate.reindex(
            w.index, method="ffill").fillna(False)

        w["FirstBreak"] = w["Close"] > w["Zone"]
        w["Rebreak"] = w["Close"] > w["Zone"]
        w["Retest"] = (w["Low"] <= w["Zone"] *
                       (1 + RETEST_TOL))
        w["Exit"] = False

        signals[t] = w

    common_idx = None
    for df in signals.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(
            df.index)

    dt = common_idx[-1]
    asof = str(dt.date())

    action = "HOLD"
    target = state["current"]
    entry_price = state["entry_price"]
    stop_price = state["stop_price"]
    equity_real = state["realized_equity"]
    equity_mtm = equity_real
    reason = "No signal."

    # =========================
    # ノーポジ
    # =========================
    if target is None:
        candidates = []

        for t, df in signals.items():
            if dt not in df.index:
                continue
            r = df.loc[dt]
            if r["WaveOK"] and r["FirstBreak"] and r["Rebreak"]:
                strength = (r["Close"] / r["Zone"] - 1.0) \
                    if not np.isnan(r["Zone"]) else -np.inf
                candidates.append({
                    "ticker": t,
                    "close": r["Close"],
                    "strength": strength,
                    "atr_pct": r["ATR_PCT"]
                })

        if candidates:
            for c in candidates:
                ap = c["atr_pct"]
                c["score"] = c["strength"] / ap \
                    if ap and ap > 0 else -np.inf

            candidates.sort(
                key=lambda x: x["score"], reverse=True)
            picked = candidates[0]

            action = "BUY"
            target = picked["ticker"]
            entry_price = picked["close"] * \
                (1 + FEE_RATE)
            stop_price = None

            state.update({
                "asof": asof,
                "current": target,
                "entry_date": asof,
                "entry_price": entry_price,
                "stop_price": stop_price
            })

            append_action({
                "date": asof,
                "action": "BUY",
                "ticker": target,
                "price": entry_price,
                "lev": LEV
            })

            reason = "New entry."

        else:
            state["asof"] = asof

    # =========================
    # 保有中
    # =========================
    else:
        df = signals[target]
        r = df.loc[dt]
        close_now = r["Close"]

        r_full = close_now / entry_price
        equity_mtm = equity_real * \
            (1 + LEV * (r_full - 1.0))

        state["asof"] = asof
        reason = "In position."

    # =========================
    # 保存
    # =========================
    save_json(STATE_JSON, state)

    signal = {
        "asof": asof,
        "action": action,
        "target": target,
        "lev": LEV,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "equity_realized": state["realized_equity"],
        "equity_mtm": equity_mtm,
        "reason": reason,
        "priority_mode": PRIORITY_MODE
    }

    save_json(SIGNAL_JSON, signal)

    print("Updated.")
    print(signal)


if __name__ == "__main__":
    main()

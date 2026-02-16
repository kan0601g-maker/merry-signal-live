# web/scripts/wave3_live_update.py
import os, json
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# CONFIG (神コード準拠)
# =========================
TICKERS = ["^GSPC", "GLD", "^N225", "VT", "SLV","XLP","XLV","XLU","IBB","SMH","QQQ","ITA"]
CASH_TICKER = "SHY"

DATA_START = "1985-01-01"
BT_START   = "2000-01-01"

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

LEV = 3.0

PRIORITY_MODE = "RISK_ADJ"
PRIORITY_LIST = ["^GSPC", "VT", "GLD", "^N225", "SLV"]

ATR_PERIOD = 14

# Output paths (Next.js public)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../web
OUT_DIR  = os.path.join(BASE_DIR, "public", "data")
STATE_JSON = os.path.join(OUT_DIR, "state.json")
SIGNAL_JSON = os.path.join(OUT_DIR, "trade_signal.json")
SNAP_JSON = os.path.join(OUT_DIR, "universe_snapshot.json")
ACTIONS_CSV = os.path.join(OUT_DIR, "actions.csv")


# =========================
# DATA HELPERS
# =========================
def download_daily(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, interval="1d", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df

def to_weekly(df_d: pd.DataFrame) -> pd.DataFrame:
    return df_d.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

def to_monthly(df_d: pd.DataFrame) -> pd.DataFrame:
    return df_d.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

def find_horizontal_wick_zone(high_arr: np.ndarray) -> float:
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

def build_wave3_monthly_gate(df_d: pd.DataFrame) -> pd.Series:
    m = to_monthly(df_d).copy()
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
        prev_low = float(np.min(lows[s:i]))
        window_high = float(np.max(highs[s:i+1]))

        if (not locked) and (lows[i] < prev_low):
            locked = True
            res = window_high
            continue

        if locked:
            if (closes[i] > res) and (closes[i] > ma200[i]):
                locked = False
                res = np.nan
                gate[i] = True
        else:
            gate[i] = (closes[i] > ma200[i])

    return pd.Series(gate, index=m.index, name="WAVE3_OK")

def map_monthly_to_weekly(m_gate: pd.Series, w_idx: pd.DatetimeIndex) -> np.ndarray:
    return m_gate.reindex(w_idx, method="ffill").fillna(False).to_numpy(dtype=bool)

def weekly_atr_pct(df_w: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = df_w["High"].to_numpy(float)
    l = df_w["Low"].to_numpy(float)
    c = df_w["Close"].to_numpy(float)

    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr, index=df_w.index).rolling(period).mean().to_numpy(float)
    return atr / c

def build_weekly_signals_for_ticker(df_d: pd.DataFrame) -> pd.DataFrame:
    df_w_full = to_weekly(df_d).copy()
    df_w = df_w_full[df_w_full.index >= BT_START].copy()
    if len(df_w) < 60:
        raise ValueError("Not enough weekly bars after BT_START")

    idx = df_w.index
    h = df_w["High"].to_numpy(float)
    l = df_w["Low"].to_numpy(float)
    c = df_w["Close"].to_numpy(float)

    zone = np.full(len(df_w), np.nan, dtype=float)
    for i in range(len(df_w)):
        zone[i] = find_horizontal_wick_zone(h[:i+1])

    first_break = (c > zone) & ~np.isnan(zone)

    retest = np.full(len(df_w), False, dtype=bool)
    for i in range(len(df_w)):
        z = zone[i]
        if np.isnan(z):
            continue
        retest[i] = (l[i] <= z * (1.0 + RETEST_TOL)) and (l[i] >= z * (1.0 - 3.0 * RETEST_TOL))

    rebreak = (c > zone) & ~np.isnan(zone)

    touch = np.full(len(df_w), False, dtype=bool)
    for i in range(len(df_w)):
        z = zone[i]
        if np.isnan(z):
            continue
        touch[i] = (abs(h[i] - z) / z <= TOUCH_TOL)

    touch_weak = np.full(len(df_w), False, dtype=bool)
    for i in range(len(df_w)):
        s = max(0, i - TOUCH_WINDOW + 1)
        cnt = int(np.sum(touch[s:i+1]))
        touch_weak[i] = (cnt < MIN_TOUCHES_IN_WINDOW)

    lower_body_break = np.full(len(df_w), False, dtype=bool)
    for i in range(len(df_w)):
        z = zone[i]
        if np.isnan(z):
            continue
        lower_body_break[i] = (c[i] < z * (1.0 - LOWER_BODY_BREAK_TOL))

    exit_signal = touch_weak & lower_body_break

    m_gate = build_wave3_monthly_gate(df_d)
    w_gate = map_monthly_to_weekly(m_gate, idx)

    atr_pct = weekly_atr_pct(df_w, ATR_PERIOD)

    out = pd.DataFrame(index=idx)
    out["High"] = h
    out["Low"] = l
    out["Close"] = c
    out["Zone"] = zone
    out["ATR_PCT"] = atr_pct
    out["FirstBreak"] = first_break
    out["Retest"] = retest
    out["Rebreak"] = rebreak
    out["Exit"] = exit_signal
    out["WaveOK"] = w_gate
    return out

def pick_candidate(candidates: list[dict]) -> dict:
    if PRIORITY_MODE == "FIXED":
        rank = {t: i for i, t in enumerate(PRIORITY_LIST)}
        candidates.sort(key=lambda x: rank.get(x["ticker"], 10**9))
        return candidates[0]

    if PRIORITY_MODE == "STRENGTH":
        candidates.sort(key=lambda x: x["strength"], reverse=True)
        return candidates[0]

    if PRIORITY_MODE == "RISK_ADJ":
        for x in candidates:
            ap = x["atr_pct"]
            x["score"] = x["strength"] / ap if (ap is not None and np.isfinite(ap) and ap > 0) else -np.inf
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[0]

    candidates.sort(key=lambda x: x["strength"], reverse=True)
    return candidates[0]


# =========================
# LIVE state
# =========================
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)

def load_state() -> dict:
    if not os.path.exists(STATE_JSON):
        return {
            "mode": "LIVE",
            "asof": None,
            "lev": LEV,
            "current": None,
            "entry_date": None,
            "entry_price": None,
            "stop_price": None,
            "realized_equity": 1.0,   # 基準1.0（ここに資金倍率を反映）
            "realized_pnl": 0.0
        }
    with open(STATE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_action(row: dict):
    header_needed = not os.path.exists(ACTIONS_CSV)
    df = pd.DataFrame([row])
    df.to_csv(ACTIONS_CSV, mode="a", index=False, header=header_needed, encoding="utf-8")

def latest_completed_week_index(common_idx: pd.DatetimeIndex) -> pd.Timestamp:
    # W-FRIの確定足を使う（最新が未確定でも resampleで確定してる前提）
    return common_idx[-1]

def compute_candidates_latest(signals_by_ticker: dict[str, pd.DataFrame], dt: pd.Timestamp) -> list[dict]:
    # 神コードと同じbreakout_seen/retest_seenの “履歴依存” は、過去バーを走査して再構築する
    candidates = []
    for t, df in signals_by_ticker.items():
        if dt not in df.index:
            continue

        breakout_seen = False
        retest_seen = False

        # 先頭から dt まで走査（stateに持たない＝壊れない）
        sub = df.loc[:dt]
        for _, r in sub.iterrows():
            if bool(r["WaveOK"]) and bool(r["FirstBreak"]):
                breakout_seen = True
            if breakout_seen and bool(r["Retest"]):
                retest_seen = True

        r = df.loc[dt]
        if bool(r["WaveOK"]) and breakout_seen and retest_seen and bool(r["Rebreak"]):
            z = float(r["Zone"])
            cc = float(r["Close"])
            atr_pct = float(r["ATR_PCT"]) if np.isfinite(r["ATR_PCT"]) else np.nan
            strength = (cc / z - 1.0) if (np.isfinite(z) and z > 0) else -np.inf
            candidates.append({
                "ticker": t,
                "close": cc,
                "zone": z,
                "atr_pct": atr_pct,
                "strength": strength
            })
    return candidates

def main():
    ensure_dirs()
    state = load_state()
    state["lev"] = LEV  # 強制固定

    # 1) signals build
    signals_by_ticker = {}
    for t in TICKERS:
        df_d = download_daily(t, DATA_START)
        sig = build_weekly_signals_for_ticker(df_d)
        signals_by_ticker[t] = sig

    # 2) cash weekly (表示用 & 参照用)
    shy_d = download_daily(CASH_TICKER, DATA_START)
    shy_w_full = to_weekly(shy_d).copy()
    shy_w = shy_w_full[shy_w_full.index >= BT_START].copy()

    # 3) 共通 index
    common_idx = None
    for df in signals_by_ticker.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    common_idx = common_idx.intersection(shy_w.index).sort_values()
    if len(common_idx) < 60:
        raise SystemExit("Common weekly index too short.")

    dt = latest_completed_week_index(common_idx)
    asof = str(dt.date())

    # 最新週のスナップ（PWA表示用）
    snap_rows = []
    for t, df in signals_by_ticker.items():
        if dt not in df.index:
            continue
        r = df.loc[dt]
        snap_rows.append({
            "ticker": t,
            "date": asof,
            "close": float(r["Close"]),
            "zone": None if np.isnan(r["Zone"]) else float(r["Zone"]),
            "wave_ok": bool(r["WaveOK"]),
            "first_break": bool(r["FirstBreak"]),
            "retest": bool(r["Retest"]),
            "rebreak": bool(r["Rebreak"]),
            "exit": bool(r["Exit"]),
            "atr_pct": None if not np.isfinite(r["ATR_PCT"]) else float(r["ATR_PCT"]),
        })

    # 4) action decision
    action = "HOLD"
    target = state.get("current")
    reason = "No signal."
    entry_price = state.get("entry_price")
    stop_price = state.get("stop_price")

    # MTM計算（表示用）：in-positionなら entry_price と close から算出
    equity_real = float(state.get("realized_equity", 1.0))
    equity_mtm = equity_real

    if target is None:
        # ノーポジ：新規エントリー検知のみ
        candidates = compute_candidates_latest(signals_by_ticker, dt)
        picked = None
        if candidates:
            picked = pick_candidate(candidates)

        if picked:
            action = "BUY"
            target = picked["ticker"]
            # エントリー価格（手数料込み）
            entry_price = picked["close"] * (1.0 + SLIPPAGE) * (1.0 + FEE_RATE)
            # “表示用ストップライン”：前週安値（PrevWeekLow）
            df = signals_by_ticker[target]
            loc = df.index.get_loc(dt)
            if loc >= 1:
                prev_low = float(df.iloc[loc - 1]["Low"])
                stop_price = prev_low
            else:
                stop_price = None

            reason = f"WaveOK + FirstBreak-seen + Retest-seen + Rebreak (picked by {PRIORITY_MODE})."

            # state更新
            state["asof"] = asof
            state["current"] = target
            state["entry_date"] = asof
            state["entry_price"] = entry_price
            state["stop_price"] = stop_price

            append_action({
                "date": asof,
                "action": "BUY",
                "ticker": target,
                "lev": LEV,
                "price": entry_price,
                "reason": reason
            })
        else:
            state["asof"] = asof

    else:
        # ポジ保有：Exit判定（神コードのExit列）
        df = signals_by_ticker[target]
        if dt in df.index:
            r = df.loc[dt]
            close_now = float(r["Close"])
            r_full_now = close_now / float(entry_price)
            equity_mtm = equity_real * (1.0 + LEV * (r_full_now - 1.0))
            if equity_mtm < 0.0:
                equity_mtm = 0.0

            if bool(r["Exit"]):
                action = "SELL"
                # 確定価格（手数料込み）
                exit_px = close_now * (1.0 - SLIPPAGE) * (1.0 - FEE_RATE)
                r_full = exit_px / float(entry_price)
                trade_r = 1.0 + LEV * (r_full - 1.0)
                if trade_r < 0.0:
                    trade_r = 0.0

                new_real = equity_real * trade_r
                pnl = new_real - equity_real

                reason = "Exit signal: touch_weak & lower_body_break (神コード)."

                # state更新（ノーポジへ）
                state["asof"] = asof
                state["realized_equity"] = float(new_real)
                state["realized_pnl"] = float(state.get("realized_pnl", 0.0) + pnl)
                state["current"] = None
                state["entry_date"] = None
                state["entry_price"] = None
                state["stop_price"] = None

                append_action({
                    "date": asof,
                    "action": "SELL",
                    "ticker": target,
                    "lev": LEV,
                    "price": exit_px,
                    "reason": reason
                })
            else:
                state["asof"] = asof
                reason = "In position. No exit."

        else:
            state["asof"] = asof
            reason = "In position, but latest week not found for current ticker."

    # 5) outputs
    save_json(STATE_JSON, state)

    signal = {
        "asof": asof,
        "action": action,
        "target": target,
        "lev": LEV,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "equity_realized": float(state.get("realized_equity", 1.0)),
        "equity_mtm": float(equity_mtm),
        "reason": reason,
        "priority_mode": PRIORITY_MODE,
    }
    save_json(SIGNAL_JSON, signal)

    save_json(SNAP_JSON, {
        "asof": asof,
        "tickers": TICKERS,
        "rows": snap_rows
    })

    print(f"Updated: {os.path.relpath(SIGNAL_JSON, BASE_DIR)} / {os.path.relpath(STATE_JSON, BASE_DIR)}")
    print(signal)

if __name__ == "__main__":
    main()

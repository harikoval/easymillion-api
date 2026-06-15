import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("TWELVEDATA_API_KEY")
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"


# ── Indicator helpers (pandas/numpy only) ─────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA: seed with mean of first `period` values, then decay."""
    arr = series.values.astype(float)
    out = np.full(len(arr), np.nan)
    # skip leading NaNs (e.g. first row of diff/shift)
    start = 0
    while start < len(arr) and np.isnan(arr[start]):
        start += 1
    if start + period > len(arr):
        return pd.Series(out, index=series.index)
    out[start + period - 1] = arr[start : start + period].mean()
    for i in range(start + period, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(out[i - 1]):
            out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return pd.Series(out, index=series.index)


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = _wilder_smooth(delta.clip(lower=0.0), period)
    avg_loss = _wilder_smooth((-delta).clip(lower=0.0), period)
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def calc_bbands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()  # ddof=1, matches TradingView / most TA tools
    return mid - num_std * std, mid + num_std * std  # lower, upper


def calc_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
               k: int = 14, smooth_k: int = 3) -> pd.Series:
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
    return raw_k.rolling(smooth_k).mean()  # smoothed %K


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14):
    ph, pl, pc = high.shift(1), low.shift(1), close.shift(1)
    tr = pd.concat([
        high - low,
        (high - pc).abs(),
        (low - pc).abs(),
    ], axis=1).max(axis=1)
    up_move = high - ph
    dn_move = pl - low
    plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)
    tr_s = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / tr_s
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _wilder_smooth(dx, period)
    return adx, plus_di, minus_di


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_candles(pair: str, outputsize: int = 100) -> pd.DataFrame:
    params = {
        "symbol": pair,
        "interval": "1min",
        "outputsize": outputsize,
        "apikey": API_KEY,
    }
    resp = requests.get(TWELVEDATA_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "error" or "values" not in data:
        raise ValueError(data.get("message", "Twelve Data error"))
    df = pd.DataFrame(data["values"])
    df = df.iloc[::-1].reset_index(drop=True)  # oldest → newest
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col])
    return df


# ── Confluence voting ─────────────────────────────────────────────────────────

def calculate_signal(df: pd.DataFrame):
    """
    5-indicator confluence vote.  Returns:
      (direction, accuracy, votes, bullish_count, bearish_count)
    direction is None when no clear signal.
    """
    close, high, low = df["close"], df["high"], df["low"]
    votes: dict = {}
    bullish = bearish = 0

    # RSI(14): <35 bullish, >65 bearish
    rsi = calc_rsi(close, 14).iloc[-1]
    if pd.notna(rsi):
        if rsi < 35:
            votes["rsi"] = "bullish"; bullish += 1
        elif rsi > 65:
            votes["rsi"] = "bearish"; bearish += 1
        else:
            votes["rsi"] = "neutral"
    else:
        votes["rsi"] = "neutral"

    # MACD(12,26,9): MACD line > signal → bullish
    ml, sl = calc_macd(close, 12, 26, 9)
    mv, sv = ml.iloc[-1], sl.iloc[-1]
    if pd.notna(mv) and pd.notna(sv):
        if mv > sv:
            votes["macd"] = "bullish"; bullish += 1
        elif mv < sv:
            votes["macd"] = "bearish"; bearish += 1
        else:
            votes["macd"] = "neutral"
    else:
        votes["macd"] = "neutral"

    # Bollinger Bands(20,2): close ≤ lower → bullish, ≥ upper → bearish
    bbl, bbu = calc_bbands(close, 20, 2.0)
    last_close = close.iloc[-1]
    lo, hi = bbl.iloc[-1], bbu.iloc[-1]
    if pd.notna(lo) and pd.notna(hi):
        if last_close <= lo:
            votes["bbands"] = "bullish"; bullish += 1
        elif last_close >= hi:
            votes["bbands"] = "bearish"; bearish += 1
        else:
            votes["bbands"] = "neutral"
    else:
        votes["bbands"] = "neutral"

    # Stochastic(14,3,3): %K < 25 → bullish, > 75 → bearish
    stoch_k = calc_stoch(high, low, close, 14, 3).iloc[-1]
    if pd.notna(stoch_k):
        if stoch_k < 25:
            votes["stoch"] = "bullish"; bullish += 1
        elif stoch_k > 75:
            votes["stoch"] = "bearish"; bearish += 1
        else:
            votes["stoch"] = "neutral"
    else:
        votes["stoch"] = "neutral"

    # ADX(14): ADX > 15 and +DI > -DI → bullish; -DI > +DI → bearish
    adx, plus_di, minus_di = calc_adx(high, low, close, 14)
    av, dmp, dmn = adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
    if pd.notna(av) and pd.notna(dmp) and pd.notna(dmn):
        if av > 15 and dmp > dmn:
            votes["adx"] = "bullish"; bullish += 1
        elif av > 15 and dmn > dmp:
            votes["adx"] = "bearish"; bearish += 1
        else:
            votes["adx"] = "neutral"
    else:
        votes["adx"] = "neutral"

    # Decision: any majority wins; no_signal only on exact tie
    if bullish > bearish:
        direction, winning = "BUY", bullish
    elif bearish > bullish:
        direction, winning = "SELL", bearish
    else:
        return None, None, votes, bullish, bearish

    accuracy = {1: 55, 2: 65, 3: 75, 4: 82, 5: 88}.get(min(winning, 5), 55)
    return direction, accuracy, votes, bullish, bearish


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/ping")
def ping():
    return jsonify({"ok": True}), 200


@app.route("/signal")
def get_signal():
    pair = request.args.get("pair", "EUR/USD")

    weekday = datetime.now(timezone.utc).weekday()  # 0=Mon … 5=Sat, 6=Sun
    if weekday >= 5:
        return jsonify({"market_closed": True, "message": "Markets are closed on weekends"})

    try:
        df = fetch_candles(pair)
        direction, accuracy, votes, bull_count, bear_count = calculate_signal(df)

        if direction is None:
            return jsonify({
                "no_signal": True,
                "pair": pair,
                "message": f"No clear signal for {pair} right now.",
            })

        entry_price = round(float(df["close"].iloc[-1]), 5)
        return jsonify({
            "signal": direction,
            "accuracy": accuracy,
            "pair": pair,
            "expiry": "1 min",
            "entry_price": entry_price,
            "votes": votes,
        })

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach Twelve Data API."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Twelve Data API timed out."}), 504
    except ValueError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route("/result")
def get_result():
    pair = request.args.get("pair", "EUR/USD")
    direction = request.args.get("direction", "BUY").upper()
    try:
        entry = float(request.args.get("entry", "0"))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid entry price"}), 400

    if direction not in ("BUY", "SELL"):
        return jsonify({"error": "direction must be BUY or SELL"}), 400

    try:
        df = fetch_candles(pair, outputsize=1)
        exit_price = round(float(df["close"].iloc[-1]), 5)
        if direction == "BUY":
            result = "WIN" if exit_price > entry else "LOSS"
        else:
            result = "WIN" if exit_price < entry else "LOSS"
        return jsonify({"result": result, "entry": entry, "exit": exit_price, "pair": pair})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug")
def debug_signal():
    pair = request.args.get("pair", "EUR/USD")
    api_key_set = bool(API_KEY)
    api_key_preview = (
        (API_KEY[:4] + "..." + API_KEY[-4:]) if api_key_set and len(API_KEY) >= 8
        else ("set" if api_key_set else "NOT SET")
    )
    try:
        df = fetch_candles(pair)
        direction, accuracy, votes, bull_count, bear_count = calculate_signal(df)
        price_range = round(float(df["close"].max() - df["close"].min()), 6)
        entry_price = round(float(df["close"].iloc[-1]), 5)
        return jsonify({
            "api_key_set": api_key_set,
            "api_key_preview": api_key_preview,
            "pair": pair,
            "candles": len(df),
            "price_range": price_range,
            "entry_price": entry_price,
            "bullish_votes": bull_count,
            "bearish_votes": bear_count,
            "votes": votes,
            "signal": direction,
            "accuracy": accuracy,
            "no_signal": direction is None,
            "warning": "Flat data likely means market is closed" if price_range < 0.0002 else None,
        })
    except Exception as e:
        return jsonify({"api_key_set": api_key_set, "api_key_preview": api_key_preview, "error": str(e)}), 500


if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("TWELVEDATA_API_KEY not set in .env")
    port = int(os.environ.get("PORT", 5000))
    print(f"Signal API running on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)

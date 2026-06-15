import os
import requests
import pandas as pd
import pandas_ta as pta
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("TWELVEDATA_API_KEY")
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"


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


def calculate_signal(df: pd.DataFrame):
    """
    5-indicator confluence voting system.
    Returns (direction, accuracy, votes, bullish_count, bearish_count).
    direction is None when no clear signal.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    votes = {}
    bullish = 0
    bearish = 0

    # ── RSI(14) ──────────────────────────────────────────────
    rsi_series = pta.rsi(close, length=14)
    rsi = rsi_series.iloc[-1] if rsi_series is not None else float("nan")
    if pd.notna(rsi):
        if rsi < 30:
            votes["rsi"] = "bullish"; bullish += 1
        elif rsi > 70:
            votes["rsi"] = "bearish"; bearish += 1
        else:
            votes["rsi"] = "neutral"
    else:
        votes["rsi"] = "neutral"

    # ── MACD(12,26,9) ────────────────────────────────────────
    macd_df = pta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        macd_line = macd_df["MACD_12_26_9"].iloc[-1]
        sig_line = macd_df["MACDs_12_26_9"].iloc[-1]
        if pd.notna(macd_line) and pd.notna(sig_line):
            if macd_line > sig_line:
                votes["macd"] = "bullish"; bullish += 1
            elif macd_line < sig_line:
                votes["macd"] = "bearish"; bearish += 1
            else:
                votes["macd"] = "neutral"
        else:
            votes["macd"] = "neutral"
    else:
        votes["macd"] = "neutral"

    # ── Bollinger Bands(20,2) ─────────────────────────────────
    bb_df = pta.bbands(close, length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        bbl = bb_df["BBL_20_2.0"].iloc[-1]
        bbu = bb_df["BBU_20_2.0"].iloc[-1]
        last_close = close.iloc[-1]
        if pd.notna(bbl) and pd.notna(bbu):
            if last_close <= bbl:
                votes["bbands"] = "bullish"; bullish += 1
            elif last_close >= bbu:
                votes["bbands"] = "bearish"; bearish += 1
            else:
                votes["bbands"] = "neutral"
        else:
            votes["bbands"] = "neutral"
    else:
        votes["bbands"] = "neutral"

    # ── Stochastic(14,3,3) ───────────────────────────────────
    stoch_df = pta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    if stoch_df is not None and not stoch_df.empty:
        stoch_k = stoch_df["STOCHk_14_3_3"].iloc[-1]
        if pd.notna(stoch_k):
            if stoch_k < 20:
                votes["stoch"] = "bullish"; bullish += 1
            elif stoch_k > 80:
                votes["stoch"] = "bearish"; bearish += 1
            else:
                votes["stoch"] = "neutral"
        else:
            votes["stoch"] = "neutral"
    else:
        votes["stoch"] = "neutral"

    # ── ADX(14) with +DI / -DI ───────────────────────────────
    adx_df = pta.adx(high, low, close, length=14)
    if adx_df is not None and not adx_df.empty:
        adx_val = adx_df["ADX_14"].iloc[-1]
        dmp = adx_df["DMP_14"].iloc[-1]
        dmn = adx_df["DMN_14"].iloc[-1]
        if pd.notna(adx_val) and pd.notna(dmp) and pd.notna(dmn):
            if adx_val > 20 and dmp > dmn:
                votes["adx"] = "bullish"; bullish += 1
            elif adx_val > 20 and dmn > dmp:
                votes["adx"] = "bearish"; bearish += 1
            else:
                votes["adx"] = "neutral"
        else:
            votes["adx"] = "neutral"
    else:
        votes["adx"] = "neutral"

    # ── Decision ─────────────────────────────────────────────
    if bullish >= 2 and bullish > bearish:
        direction = "BUY"
        winning = bullish
    elif bearish >= 2 and bearish > bullish:
        direction = "SELL"
        winning = bearish
    else:
        return None, None, votes, bullish, bearish

    accuracy = {2: 60, 3: 70, 4: 80, 5: 90}.get(min(winning, 5), 60)
    return direction, accuracy, votes, bullish, bearish


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

        return jsonify({
            "result": result,
            "entry": entry,
            "exit": exit_price,
            "pair": pair,
        })
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
        price_range = round(df["close"].max() - df["close"].min(), 6)
        unique_closes = int(df["close"].nunique())
        entry_price = round(float(df["close"].iloc[-1]), 5)
        return jsonify({
            "api_key_set": api_key_set,
            "api_key_preview": api_key_preview,
            "pair": pair,
            "candles": len(df),
            "price_range": price_range,
            "unique_closes": unique_closes,
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

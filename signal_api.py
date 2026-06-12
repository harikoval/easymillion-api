import os
import requests
import pandas as pd
import ta
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("TWELVEDATA_API_KEY")
TWELVEDATA_URL = "https://api.twelvedata.com/time_series"


def fetch_candles(pair: str) -> pd.DataFrame:
    params = {
        "symbol": pair,
        "interval": "1min",
        "outputsize": 50,
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
    close = df["close"]

    rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_obj = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    bb_obj = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)

    df["RSI"] = rsi_series
    df["MACD"] = macd_obj.macd()
    df["MACD_signal"] = macd_obj.macd_signal()
    df["BB_lower"] = bb_obj.bollinger_lband()
    df["BB_upper"] = bb_obj.bollinger_hband()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = last["RSI"]
    macd_now = last["MACD"]
    sig_now = last["MACD_signal"]
    macd_prev = prev["MACD"]
    sig_prev = prev["MACD_signal"]
    close = last["close"]
    bb_lower = last["BB_lower"]
    bb_upper = last["BB_upper"]

    macd_crossed_up = macd_prev <= sig_prev and macd_now > sig_now
    macd_crossed_down = macd_prev >= sig_prev and macd_now < sig_now

    # Determine direction
    if rsi < 35 and macd_crossed_up:
        direction = "CALL"
    elif rsi > 65 and macd_crossed_down:
        direction = "PUT"
    elif macd_crossed_up:
        direction = "CALL"
    elif macd_crossed_down:
        direction = "PUT"
    elif rsi < 50:
        direction = "CALL"
    else:
        direction = "PUT"

    # Accuracy: count how many indicators agree (0–4 points → 65–89%)
    score = 0

    if direction == "CALL":
        if rsi < 35:
            score += 2
        elif rsi < 50:
            score += 1
        if macd_crossed_up:
            score += 1
        elif macd_now > sig_now:
            score += 0.5
        if pd.notna(bb_lower) and close <= bb_lower:
            score += 1
    else:
        if rsi > 65:
            score += 2
        elif rsi > 50:
            score += 1
        if macd_crossed_down:
            score += 1
        elif macd_now < sig_now:
            score += 0.5
        if pd.notna(bb_upper) and close >= bb_upper:
            score += 1

    accuracy = int(65 + min(score / 4, 1) * 24)
    return direction, accuracy


@app.route("/")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/signal")
def get_signal():
    pair = request.args.get("pair", "EUR/USD")

    try:
        df = fetch_candles(pair)
        direction, accuracy = calculate_signal(df)
        return jsonify({
            "signal": direction,
            "accuracy": accuracy,
            "pair": pair,
            "expiry": "1 min",
        })

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach Twelve Data API. Check your internet connection."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Twelve Data API timed out."}), 504
    except ValueError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("TWELVEDATA_API_KEY not set in .env")
    port = int(os.environ.get("PORT", 5000))
    print(f"Signal API running on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)

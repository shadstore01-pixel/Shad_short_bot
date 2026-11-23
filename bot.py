import os
import time
import threading
import requests
import pandas as pd
from http.server import BaseHTTPRequestHandler, HTTPServer

# ====== SETTINGS ======
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

TIMEFRAME = "1h"
CHECK_EVERY = 300          # seconds to wait after each full scan
MIN_VOLUME = 5_000_000     # minimum 24h volume in USDT (set 0 for all coins)
DELAY = 0.25               # pause between coins (avoids Binance rate limit)

BASE = "https://fapi.binance.com"
last_alert = {}
stats = {"alerts": 0}


# ====== TELEGRAM ======
def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
    except Exception as e:
        print("Telegram error:", e)


# ====== ALL SYMBOLS ======
def get_all_symbols():
    info = requests.get(f"{BASE}/fapi/v1/exchangeInfo", timeout=20).json()
    allowed = {
        s["symbol"]
        for s in info["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
    }
    tickers = requests.get(f"{BASE}/fapi/v1/ticker/24hr", timeout=20).json()
    result = []
    for t in tickers:
        if t["symbol"] in allowed and float(t["quoteVolume"]) >= MIN_VOLUME:
            result.append(t["symbol"])
    return sorted(result)


# ====== DATA ======
def get_klines(symbol):
    r = requests.get(
        f"{BASE}/fapi/v1/klines",
        params={"symbol": symbol, "interval": TIMEFRAME, "limit": 300},
        timeout=15,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df.iloc[:-1].reset_index(drop=True)  # drop the unfinished candle


def get_oi_trend(symbol):
    r = requests.get(
        f"{BASE}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": "1h", "limit": 3},
        timeout=15,
    )
    r.raise_for_status()
    d = r.json()
    first = float(d[0]["sumOpenInterest"])
    last = float(d[-1]["sumOpenInterest"])
    change = (last - first) / first * 100
    return last, change, change > 0


def get_top_ls_ratio(symbol):
    r = requests.get(
        f"{BASE}/futures/data/topLongShortAccountRatio",
        params={"symbol": symbol, "period": "1h", "limit": 1},
        timeout=15,
    )
    r.raise_for_status()
    d = r.json()[0]
    return float(d["longShortRatio"]), float(d["longAccount"]), float(d["shortAccount"])


# ====== INDICATORS ======
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def stoch_rsi(s, rsi_len=14, stoch_len=14, k=3, d=3):
    r = rsi(s, rsi_len)
    lo = r.rolling(stoch_len).min()
    hi = r.rolling(stoch_len).max()
    st = (r - lo) / (hi - lo) * 100
    kk = st.rolling(k).mean()
    dd = kk.rolling(d).mean()
    return kk, dd


def parabolic_sar(high, low, af_step=0.02, af_max=0.2):
    h = high.values
    l = low.values
    n = len(h)
    sar = [0.0] * n
    uptrend = True
    af = af_step
    ep = h[0]
    sar[0] = l[0]
    for i in range(1, n):
        prev = sar[i - 1]
        new = prev + af * (ep - prev)
        if uptrend:
            new = min(new, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < new:
                uptrend = False
                new = ep
                ep = l[i]
                af = af_step
            elif h[i] > ep:
                ep = h[i]
                af = min(af + af_step, af_max)
        else:
            new = max(new, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > new:
                uptrend = True
                new = ep
                ep = h[i]
                af = af_step
            elif l[i] < ep:
                ep = l[i]
                af = min(af + af_step, af_max)
        sar[i] = new
    return pd.Series(sar, index=high.index)


# ====== MARKET CHECK ======
def check_symbol(symbol):
    df = get_klines(symbol)
    if len(df) < 210:  # new coin, not enough data for EMA200
        return
    close = df["close"]
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi"] = rsi(close)
    df["stk"], df["std"] = stoch_rsi(close)
    df["sar"] = parabolic_sar(df["high"], df["low"])

    row = df.iloc[-1]
    candle_time = int(row["time"])
    price = row["close"]

    # Check price conditions first (cheap). Fetch OI / Long-Short only if they match.
    bull_tech = price > row["ema20"] and price > row["sar"] and row["stk"] < 20

    level_name = ""
    for name in ["ema50", "ema200"]:
        lvl = row[name]
        if row["high"] >= lvl and price < lvl:
            level_name = name.upper()
            break
    bear_tech = (
        level_name != ""
        and price < row["ema20"]
        and price < row["sar"]
    )

    if not bull_tech and not bear_tech:
        return

    oi, oi_change, oi_rising = get_oi_trend(symbol)
    if not oi_rising:
        return
    ls_ratio, long_acc, short_acc = get_top_ls_ratio(symbol)

    common = (
        f"💰 Price: {price}\n"
        f"📈 EMA20: {row['ema20']:.5f}\n"
        f"📈 EMA50: {row['ema50']:.5f}\n"
        f"📈 EMA200: {row['ema200']:.5f}\n"
        f"🎯 Parabolic SAR: {row['sar']:.5f}\n"
        f"📊 RSI: {row['rsi']:.2f}\n"
        f"⚡ StochRSI K: {row['stk']:.2f} | D: {row['std']:.2f}\n"
        f"📦 Open Interest: {oi:,.0f} ({oi_change:+.2f}% in last 3h)\n"
        f"⚖️ Top Trader Long/Short: {ls_ratio:.2f} "
        f"(Long {long_acc*100:.1f}% / Short {short_acc*100:.1f}%)"
    )

    if bull_tech and last_alert.get((symbol, "bull")) != candle_time:
        last_alert[(symbol, "bull")] = candle_time
        stats["alerts"] += 1
        send_telegram(f"🟢 BULLISH PULLBACK ALERT ({symbol}, {TIMEFRAME})\n\n{common}")

    if bear_tech and last_alert.get((symbol, "bear")) != candle_time:
        last_alert[(symbol, "bear")] = candle_time
        stats["alerts"] += 1
        send_telegram(
            f"🔴 BEARISH WARNING ({symbol}, {TIMEFRAME})\n"
            f"Price rejected from {level_name}, Open Interest rising\n\n{common}"
        )


def bot_loop():
    send_telegram("🤖 Bot started! Scanning all Binance USDT futures.")
    first_scan = True
    warned = False
    while True:
        try:
            symbols = get_all_symbols()
            print(f"Total coins: {len(symbols)}", flush=True)
        except Exception as e:
            print("Symbol list error:", e, flush=True)
            if not warned:
                warned = True
                send_telegram(f"⚠️ Could not get coin list from Binance: {str(e)[:200]}")
            time.sleep(60)
            continue

        errors = 0
        last_error = ""
        alerts_before = stats["alerts"]
        for symbol in symbols:
            try:
                check_symbol(symbol)
            except Exception as e:
                errors += 1
                last_error = f"{symbol}: {e}"
                print(f"Error {symbol}: {e}", flush=True)
                if "429" in str(e) or "418" in str(e):
                    time.sleep(60)  # Binance asked us to slow down
            time.sleep(DELAY)

        new_alerts = stats["alerts"] - alerts_before
        print(f"Scan done. Coins: {len(symbols)}, errors: {errors}, alerts: {new_alerts}", flush=True)

        if first_scan:
            first_scan = False
            msg = f"✅ First scan done. Coins checked: {len(symbols)}, alerts: {new_alerts}, errors: {errors}"
            if errors:
                msg += f"\nLast error: {last_error[:200]}"
            send_telegram(msg)

        time.sleep(CHECK_EVERY)


# ====== TINY WEB SERVER (keeps Render awake) ======
class Ping(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=bot_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    HTTPServer(("0.0.0.0", port), Ping).serve_forever()

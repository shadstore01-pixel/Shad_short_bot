import os
import re
import math
import time
import threading
import requests
import pandas as pd
from http.server import BaseHTTPRequestHandler, HTTPServer


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# ====== SETTINGS (can be changed in Render > Environment) ======
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

TIMEFRAME = "1h"
MIN_VOLUME = env_float("MIN_VOLUME", 5_000_000)         # min 24h volume in USDT
DELAY = 0.2                                             # pause between coins
SCAN_EVERY_MINUTES = int(env_float("SCAN_EVERY_MINUTES", 15))  # how often to scan
USE_LIVE_CANDLE = os.environ.get("USE_LIVE_CANDLE", "1") == "1"  # 1 = use the forming candle
REQUIRE_UPTREND = os.environ.get("REQUIRE_UPTREND", "1") == "1"  # LONG needs EMA50 > EMA200

MAX_ALERTS_PER_HOUR = int(env_float("MAX_ALERTS_PER_HOUR", 3))   # about 70 per day at most
MAX_LONG_PER_SCAN = int(env_float("MAX_LONG_PER_SCAN", 2))
MAX_SHORT_PER_SCAN = int(env_float("MAX_SHORT_PER_SCAN", 1))
MIN_LONG_SCORE = int(env_float("MIN_LONG_SCORE", 55))
MIN_SHORT_SCORE = int(env_float("MIN_SHORT_SCORE", 60))

# Trade plan
ACCOUNT_BALANCE = env_float("ACCOUNT_BALANCE", 100)     # your trading balance in USDT
RISK_PERCENT = env_float("RISK_PERCENT", 1)             # % of balance risked per trade (NORMAL)
MAX_LEVERAGE = int(env_float("MAX_LEVERAGE", 20))
MAX_MARGIN_PERCENT = env_float("MAX_MARGIN_PERCENT", 25)
SL_ATR_MULT = env_float("SL_ATR_MULT", 1.5)             # stop loss = 1.5 x ATR
TP1_R = env_float("TP1_R", 1.5)
TP2_R = env_float("TP2_R", 3)

BASE = "https://fapi.binance.com"
last_alert = {}
stats = {"alerts": 0}
hour_state = {"hour": -1, "total": 0, "short": 0}


# ====== TELEGRAM ======
def send_telegram(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
    except Exception as e:
        print("Telegram error:", e, flush=True)


# ====== ALL SYMBOLS ======
def get_all_symbols():
    resp = requests.get(f"{BASE}/fapi/v1/exchangeInfo", timeout=20)
    try:
        info = resp.json()
    except Exception:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:150]}")
    if "symbols" not in info:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:150]}")
    allowed = {
        s["symbol"]
        for s in info["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
    }
    resp2 = requests.get(f"{BASE}/fapi/v1/ticker/24hr", timeout=20)
    try:
        tickers = resp2.json()
    except Exception:
        tickers = None
    if not isinstance(tickers, list):
        if not stats.get("ticker_warned"):
            stats["ticker_warned"] = True
            send_telegram(
                f"⚠️ Volume filter skipped. Binance said: HTTP {resp2.status_code}: {resp2.text[:150]}"
            )
        return sorted(allowed)
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
    if not USE_LIVE_CANDLE:
        df = df.iloc[:-1]  # drop the unfinished candle
    return df.reset_index(drop=True)


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


# ====== TRADE PLAN ======
def fmt_price(p):
    if p >= 100:
        return f"{p:.2f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.5f}"
    return f"{p:.8f}"


def size_for(risk_pct, sl_pct):
    """Position size, leverage, margin and real risk for a given risk % of balance."""
    risk_amount = ACCOUNT_BALANCE * risk_pct / 100
    notional = risk_amount / sl_pct
    max_margin = ACCOUNT_BALANCE * MAX_MARGIN_PERCENT / 100
    lev = max(1, math.ceil(notional / max_margin))
    if lev > MAX_LEVERAGE:
        lev = MAX_LEVERAGE
        notional = max_margin * lev
    margin = notional / lev
    real_risk = notional * sl_pct
    return notional, lev, margin, real_risk


def trade_plan(direction, entry, atr_value):
    risk_dist = SL_ATR_MULT * atr_value
    if direction == "LONG":
        sl = entry - risk_dist
        tp1 = entry + TP1_R * risk_dist
        tp2 = entry + TP2_R * risk_dist
    else:
        sl = entry + risk_dist
        tp1 = entry - TP1_R * risk_dist
        tp2 = entry - TP2_R * risk_dist
    if sl <= 0 or tp1 <= 0 or tp2 <= 0 or risk_dist <= 0:
        return "📋 Trade plan not available (volatility too high for this coin)."

    sl_pct = risk_dist / entry
    text = (
        f"📋 TRADE PLAN ({direction})\n"
        f"Entry: ~{fmt_price(entry)}\n"
        f"Stop Loss: {fmt_price(sl)} ({sl_pct*100:.2f}% from entry)\n"
        f"TP1: {fmt_price(tp1)}\n"
        f"TP2: {fmt_price(tp2)}\n\n"
        f"Choose your size (balance {ACCOUNT_BALANCE:g} USDT):"
    )
    levels = (
        ("🟢 LOW", RISK_PERCENT * 0.5),
        ("🟡 NORMAL", RISK_PERCENT),
        ("🔴 HIGH", RISK_PERCENT * 2),
    )
    warn = False
    for name, rp in levels:
        notional, lev, margin, real_risk = size_for(rp, sl_pct)
        text += (
            f"\n\n{name} (risk {rp:g}%)\n"
            f"Position: {notional:.2f} USDT | Leverage: {lev}x | Margin: {margin:.2f} USDT\n"
            f"Loss at SL: -{real_risk:.2f} USDT\n"
            f"Profit: TP1 +{real_risk*TP1_R:.2f} | TP2 +{real_risk*TP2_R:.2f} USDT"
        )
        if sl_pct * lev > 0.8:
            warn = True
    if warn:
        text += "\n\n⚠️ At some sizes the stop loss is close to liquidation. Use lower leverage."
    text += (
        "\n\nYou can enter with less or more. Loss at SL depends on POSITION size, "
        "not on leverage. Higher leverage only lowers the margin and moves liquidation closer."
        "\nThis is a plan, not financial advice. Always set the stop loss."
    )
    return text


# ====== SCORING ======
def long_score(price, ema20, ema50, ema200, atr_v, stk, std, rsi_v):
    score = 0.0
    if ema50 > ema200:
        score += 15                       # bigger trend is up
    if ema20 > ema50:
        score += 10                       # short trend is up too
    score += max(0.0, min(30.0, 30.0 - stk))   # deeper oversold = better
    if not pd.isna(std) and stk > std:
        score += 15                       # StochRSI turning up
    if 40 <= rsi_v <= 65:
        score += 10                       # pullback not broken
    if atr_v > 0 and 0 <= (price - ema20) / atr_v <= 1.5:
        score += 10                       # entry close to EMA20 = better risk/reward
    return int(round(score))


def short_score(ema20, ema50, ema200, stk, rsi_v):
    score = 40.0
    if ema50 < ema200:
        score += 20
    if ema20 < ema50:
        score += 15
    if rsi_v < 50:
        score += 10
    if not pd.isna(stk) and stk > 60:
        score += 15
    return int(round(score))


# ====== MARKET CHECK ======
def analyze_symbol(symbol):
    """Cheap price-based check. Returns a list of candidate signals (no OI yet)."""
    df = get_klines(symbol)
    if len(df) < 210:  # new coin, not enough data for EMA200
        return []
    close = df["close"]
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi"] = rsi(close)
    df["stk"], df["std"] = stoch_rsi(close)
    df["sar"] = parabolic_sar(df["high"], df["low"])
    prev_close = close.shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    row = df.iloc[-1]
    price = float(row["close"])
    candle_time = int(row["time"])
    atr_v = float(row["atr"])
    ema20, ema50, ema200 = float(row["ema20"]), float(row["ema50"]), float(row["ema200"])
    sar, rsi_v = float(row["sar"]), float(row["rsi"])
    stk, std = row["stk"], row["std"]

    base = {
        "symbol": symbol, "price": price, "atr": atr_v, "candle_time": candle_time,
        "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "sar": sar, "rsi": rsi_v, "stk": stk, "std": std,
    }
    out = []

    # --- LONG: pullback in an uptrend ---
    uptrend_ok = ema50 > ema200 or not REQUIRE_UPTREND
    if (
        not pd.isna(stk)
        and price > ema20
        and price > sar
        and stk < 30
        and uptrend_ok
        and last_alert.get((symbol, "LONG")) != candle_time
    ):
        c = dict(base, direction="LONG", level="")
        c["score"] = long_score(price, ema20, ema50, ema200, atr_v, float(stk), std, rsi_v)
        if c["score"] >= MIN_LONG_SCORE:
            out.append(c)

    # --- SHORT: rejection from EMA50 / EMA200 ---
    level_name = ""
    for name, lvl in (("EMA50", ema50), ("EMA200", ema200)):
        if float(row["high"]) >= lvl and price < lvl:
            level_name = name
            break
    if (
        level_name
        and price < ema20
        and price < sar
        and last_alert.get((symbol, "SHORT")) != candle_time
    ):
        c = dict(base, direction="SHORT", level=level_name)
        c["score"] = short_score(ema20, ema50, ema200, stk, rsi_v)
        if c["score"] >= MIN_SHORT_SCORE:
            out.append(c)

    return out


def build_message(c, oi, oi_change, ls_ratio, long_acc, short_acc):
    stk_txt = f"{c['stk']:.2f}" if not pd.isna(c["stk"]) else "n/a"
    std_txt = f"{c['std']:.2f}" if not pd.isna(c["std"]) else "n/a"
    common = (
        f"💰 Price: {fmt_price(c['price'])}\n"
        f"📈 EMA20: {c['ema20']:.5f}\n"
        f"📈 EMA50: {c['ema50']:.5f}\n"
        f"📈 EMA200: {c['ema200']:.5f}\n"
        f"🎯 Parabolic SAR: {c['sar']:.5f}\n"
        f"📊 RSI: {c['rsi']:.2f}\n"
        f"⚡ StochRSI K: {stk_txt} | D: {std_txt}\n"
        f"📦 Open Interest: {oi:,.0f} ({oi_change:+.2f}% in last 3h)\n"
        f"⚖️ Top Trader Long/Short: {ls_ratio:.2f} "
        f"(Long {long_acc*100:.1f}% / Short {short_acc*100:.1f}%)"
    )
    live = "⏱ Candle still forming (signal may change)\n" if USE_LIVE_CANDLE else ""
    if c["direction"] == "LONG":
        head = f"🟢 LONG SIGNAL ({c['symbol']}, {TIMEFRAME}) | Score {c['score']}/100\n{live}\n"
    else:
        head = (
            f"🔴 SHORT WARNING ({c['symbol']}, {TIMEFRAME}) | Score {c['score']}/100\n"
            f"Price rejected from {c['level']}, Open Interest rising\n{live}\n"
        )
    return head + common + "\n\n" + trade_plan(c["direction"], c["price"], c["atr"])


def pick_and_send(candidates):
    """Rank candidates, check Open Interest on the best ones, and send the top signals."""
    hour_key = int(time.time() // 3600)
    if hour_state["hour"] != hour_key:
        hour_state.update({"hour": hour_key, "total": 0, "short": 0})

    longs = sorted((c for c in candidates if c["direction"] == "LONG"), key=lambda c: -c["score"])
    shorts = sorted((c for c in candidates if c["direction"] == "SHORT"), key=lambda c: -c["score"])

    sent = 0
    lookups = 0
    for group, limit, direction in (
        (longs, MAX_LONG_PER_SCAN, "LONG"),
        (shorts, MAX_SHORT_PER_SCAN, "SHORT"),
    ):
        picked = 0
        for c in group:
            if picked >= limit or hour_state["total"] >= MAX_ALERTS_PER_HOUR:
                break
            if direction == "SHORT" and hour_state["short"] >= 1:
                break
            if lookups >= 15:
                break
            lookups += 1
            try:
                oi, oi_change, rising = get_oi_trend(c["symbol"])
                if not rising:
                    continue
                ls_ratio, long_acc, short_acc = get_top_ls_ratio(c["symbol"])
            except Exception as e:
                print(f"OI error {c['symbol']}: {e}", flush=True)
                if "429" in str(e) or "418" in str(e):
                    time.sleep(300)
                    return sent
                continue
            send_telegram(build_message(c, oi, oi_change, ls_ratio, long_acc, short_acc))
            last_alert[(c["symbol"], direction)] = c["candle_time"]
            picked += 1
            sent += 1
            hour_state["total"] += 1
            if direction == "SHORT":
                hour_state["short"] += 1
            stats["alerts"] += 1
            time.sleep(0.5)
    return sent


def bot_loop():
    send_telegram("🤖 Bot started! Scanning all Binance USDT futures.")
    first_scan = True
    warned = False
    while True:
        try:
            symbols = get_all_symbols()
            print(f"Total coins: {len(symbols)}", flush=True)
            warned = False
        except Exception as e:
            print("Symbol list error:", e, flush=True)
            if not warned:
                warned = True
                send_telegram(f"⚠️ Could not get coin list from Binance: {str(e)[:200]}")
            wait = 60
            m = re.search(r"banned until (\d+)", str(e))
            if m:
                remaining = int(m.group(1)) / 1000 - time.time()
                wait = max(60, min(remaining + 10, 86400))
            time.sleep(wait)
            continue

        errors = 0
        last_error = ""
        candidates = []
        for symbol in symbols:
            try:
                candidates.extend(analyze_symbol(symbol))
            except Exception as e:
                errors += 1
                last_error = f"{symbol}: {e}"
                print(f"Error {symbol}: {e}", flush=True)
                if "429" in str(e) or "418" in str(e):
                    time.sleep(300)  # Binance asked us to slow down
                    break
            time.sleep(DELAY)

        sent = pick_and_send(candidates)
        print(
            f"Scan done. Coins: {len(symbols)}, candidates: {len(candidates)}, "
            f"sent: {sent}, errors: {errors}",
            flush=True,
        )

        if first_scan:
            first_scan = False
            msg = (
                f"✅ First scan done. Coins checked: {len(symbols)}, "
                f"candidates: {len(candidates)}, signals sent: {sent}, errors: {errors}"
            )
            if errors:
                msg += f"\nLast error: {last_error[:200]}"
            send_telegram(msg)

        interval = max(5, SCAN_EVERY_MINUTES) * 60
        time.sleep(interval - (time.time() % interval) + 20)


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

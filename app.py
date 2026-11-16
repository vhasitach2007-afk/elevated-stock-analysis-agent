import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3, json
from datetime import datetime
from textblob import TextBlob
import plotly.graph_objects as go
from pathlib import Path

DB = Path(__file__).with_name("stock_analysis.db")

def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, created_at TEXT, result_json TEXT
        )""")

def save_result(symbol, result):
    with sqlite3.connect(DB) as conn:
        conn.execute("INSERT INTO analyses(symbol, created_at, result_json) VALUES(?,?,?)",
                     (symbol, datetime.now().isoformat(timespec="seconds"),
                      json.dumps(result, default=str)))

def market_data_agent(symbol, period):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval="1d", auto_adjust=True)
    if hist.empty:
        raise ValueError("No data returned. Check the ticker symbol.")
    info = {}
    try:
        raw = ticker.info or {}
        info = {k: raw.get(k) for k in ["longName", "sector", "industry", "marketCap", "trailingPE"]}
        info = {k:v for k,v in info.items() if v is not None}
    except Exception:
        pass
    return hist, info

def technical_agent(hist):
    close = hist["Close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
    typical = (hist["High"] + hist["Low"] + close) / 3
    vwap = (typical * hist["Volume"]).cumsum() / hist["Volume"].cumsum().replace(0, np.nan)
    last = {
        "Latest close": float(close.iloc[-1]),
        "RSI (14)": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
        "MACD": float(macd.iloc[-1]),
        "MACD signal": float(signal.iloc[-1]),
        "SMA 20": float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None,
        "SMA 50": float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None,
        "VWAP": float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else None
    }
    notes = []
    if last["RSI (14)"] is not None:
        notes.append("RSI is relatively high" if last["RSI (14)"] >= 70 else "RSI is relatively low" if last["RSI (14)"] <= 30 else "RSI is in a middle range")
    notes.append("MACD is above its signal line" if last["MACD"] > last["MACD signal"] else "MACD is below its signal line")
    if last["SMA 20"] is not None:
        notes.append("Price is above its 20-day average" if last["Latest close"] > last["SMA 20"] else "Price is below its 20-day average")
    return last, notes

def sentiment_agent(lines):
    lines = [x.strip() for x in lines if x.strip()]
    if not lines:
        return {"label":"Not available", "score":None, "headlines":[]}
    score = float(np.mean([TextBlob(x).sentiment.polarity for x in lines]))
    label = "Positive tone" if score > .1 else "Negative tone" if score < -.1 else "Neutral or mixed tone"
    return {"label":label, "score":round(score,3), "headlines":lines}

def prediction_agent(hist):
    returns = hist["Close"].pct_change().dropna()
    if len(returns) < 30:
        return {"state":"Insufficient data", "probabilities":{}, "note":"At least 30 daily returns are needed."}
    states = np.where(returns < -.005, 0, np.where(returns > .005, 2, 1))
    names = ["Down", "Flat", "Up"]
    counts = np.zeros((3,3))
    for a,b in zip(states[:-1], states[1:]):
        counts[int(a),int(b)] += 1
    sums = counts.sum(axis=1, keepdims=True)
    trans = np.divide(counts, sums, out=np.zeros_like(counts), where=sums != 0)
    current = int(states[-1])
    return {"state":names[current],
            "probabilities":{names[i]:round(float(trans[current,i]),3) for i in range(3)},
            "note":"Historical transition frequencies only; not a reliable forecast."}

def risk_agent(hist):
    returns = hist["Close"].pct_change().dropna()
    if len(returns) < 2:
        return {"volatility":None, "drawdown":None}
    wealth = (1 + returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    return {"volatility":float(returns.std()*np.sqrt(252)), "drawdown":float(dd.min())}

st.set_page_config(page_title="StockInsight AI", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp{background:#0b1020;color:#edf2ff}
[data-testid="stSidebar"]{background:#111a2e}
.block-container{max-width:1400px;padding-top:2rem}
.hero{background:linear-gradient(120deg,#172b4d,#25204b);padding:25px;border-radius:18px;border:1px solid #33466c;margin-bottom:20px}
.kicker{color:#9fb8ef;letter-spacing:2px;font-size:12px;font-weight:bold}
.hero h1{color:white!important;margin:5px 0}
.hero p{color:#c1cde5}
div[data-testid="stMetric"]{background:#141f35;border:1px solid #293957;padding:14px;border-radius:12px}
</style>
""", unsafe_allow_html=True)
init_db()

with st.sidebar:
    st.title("📈 StockInsight AI")
    st.caption("MULTI-AGENT STOCK RESEARCH")
    symbol = st.text_input("Stock ticker", "RELIANCE.NS").strip().upper()
    period = st.selectbox("Historical period", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
    news_text = st.text_area("News headlines (one per line)", height=140,
                             placeholder="Company reports higher quarterly revenue")
    run = st.button("Run stock analysis", type="primary", use_container_width=True)
    st.caption("Examples: RELIANCE.NS, TCS.NS, INFY.NS, AAPL, MSFT")

st.markdown('<div class="hero"><div class="kicker">MULTI-AGENT STOCK ANALYSIS PLATFORM</div><h1>Elevated Stock Analysis Agent</h1><p>Market data, technical indicators, sentiment, Markov probabilities and historical risk in one dashboard.</p></div>', unsafe_allow_html=True)

if run:
    try:
        with st.spinner("Running the analysis agents..."):
            hist, company = market_data_agent(symbol, period)
            technical, notes = technical_agent(hist)
            sentiment = sentiment_agent(news_text.splitlines())
            prediction = prediction_agent(hist)
            risk = risk_agent(hist)
            result = {"company":company,"technical":technical,"notes":notes,
                      "sentiment":sentiment,"prediction":prediction,"risk":risk}
            save_result(symbol, result)
            st.session_state["result"] = result
            st.session_state["hist"] = hist
            st.session_state["symbol"] = symbol
    except Exception as e:
        st.error(f"Analysis could not be completed: {e}")
        st.info("Check the ticker symbol and try again. Market-data services may occasionally be unavailable.")

if "result" in st.session_state:
    r = st.session_state["result"]
    hist = st.session_state["hist"]
    sym = st.session_state["symbol"]
    t, risk, sent, pred = r["technical"], r["risk"], r["sentiment"], r["prediction"]
    st.subheader(r["company"].get("longName", sym))
    st.caption(f"Ticker: {sym}")
    cols = st.columns(4)
    cols[0].metric("Latest close", f"{t['Latest close']:,.2f}")
    cols[1].metric("RSI (14)", f"{t['RSI (14)']:.2f}" if t["RSI (14)"] is not None else "N/A")
    cols[2].metric("MACD", f"{t['MACD']:.3f}")
    cols[3].metric("Annualized volatility", f"{risk['volatility']:.1%}" if risk["volatility"] is not None else "N/A")

    left,right = st.columns([1.6,1])
    with left:
        st.subheader("Stock price performance")
        fig = go.Figure(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"],
                                      low=hist["Low"], close=hist["Close"], name="Price"))
        fig.update_layout(template="plotly_dark",height=430,xaxis_rangeslider_visible=False,
                          paper_bgcolor="#0b1020",plot_bgcolor="#111a2e")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Agent status")
        for name,detail in [
            ("Market Data Agent","Historical prices retrieved"),
            ("Technical Analysis Agent","RSI, MACD, SMA and VWAP calculated"),
            ("News Sentiment Agent",sent["label"]),
            ("Prediction Agent",f"Latest state: {pred['state']}"),
            ("Risk Analysis Agent","Volatility and drawdown calculated"),
            ("Report Agent","Combined summary prepared")]:
            st.success(f"**{name}**\\n\\n{detail}")

    st.divider()
    tab1,tab2,tab3,tab4 = st.tabs(["Technical analysis","News sentiment","Markov prediction","Risk analysis"])
    with tab1:
        st.json(t)
        for n in r["notes"]: st.write("•",n)
    with tab2:
        st.metric("Headline tone",sent["label"])
        if sent["score"] is not None: st.metric("Sentiment score",sent["score"])
        if sent["headlines"]:
            for h in sent["headlines"]: st.write("•",h)
        else: st.info("Add headlines in the sidebar to run sentiment analysis.")
    with tab3:
        st.write("Latest historical state:",pred["state"])
        if pred["probabilities"]:
            st.bar_chart(pd.Series(pred["probabilities"],name="Transition probability"))
        st.caption(pred["note"])
    with tab4:
        a,b=st.columns(2)
        a.metric("Annualized volatility",f"{risk['volatility']:.2%}" if risk["volatility"] is not None else "N/A")
        b.metric("Maximum drawdown",f"{risk['drawdown']:.2%}" if risk["drawdown"] is not None else "N/A")
        st.caption("These describe historical risk and do not predict future losses.")

    st.subheader("Final analysis report")
    st.write(f"Analysis generated for **{sym}**.")
    st.write(f"Technical observations: {', '.join(r['notes'])}.")
    st.write(f"News tone: **{sent['label']}**.")
    st.write(f"Markov model's latest historical state: **{pred['state']}**.")
    st.warning("Educational information only. Not financial advice. Historical patterns and model probabilities do not guarantee future performance.")
    export = hist.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download historical data CSV",export,f"{sym}_history.csv","text/csv")

st.divider()
st.subheader("Saved analysis history")
with sqlite3.connect(DB) as conn:
    saved = pd.read_sql_query("SELECT id, symbol, created_at FROM analyses ORDER BY id DESC LIMIT 15",conn)
if saved.empty: st.caption("No saved analyses yet.")
else: st.dataframe(saved,use_container_width=True,hide_index=True)

st.caption("Educational prototype. Data may be delayed or incomplete. Not investment advice.")

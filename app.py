import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=APP_DIR / ".env")
    except ImportError:
        pass
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"   # collapsed by default — cleaner on load
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1b2a 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c9d8e8 !important; }
.stApp { background: #07111f; color: #c9d8e8; }

.insight-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #122338 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.risk-flag {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d1010 100%);
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    color: #fca5a5;
}
.ok-flag {
    background: linear-gradient(135deg, #041a10 0%, #062b18 100%);
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    color: #86efac;
}
.metric-box {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.9rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-box .label { font-size: 0.72rem; color: #6b8fad; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-box .value { font-size: 1.4rem; font-family: 'Space Mono', monospace; color: #38bdf8; font-weight: 700; }

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-weight: 700;
    letter-spacing: 0.04em; padding: 0.55rem 1.6rem; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #3b82f6);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(14,165,233,0.35);
}
[data-baseweb="tab-list"] {
    background: #0d1b2a;
    border-radius: 10px;
    gap: 3px;
    padding: 4px;
    flex-wrap: wrap;
}
[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6b8fad;
    border-radius: 7px;
    padding: 0.4rem 0.8rem;
}
[aria-selected="true"][data-baseweb="tab"] { background: #1e3a5f; color: #38bdf8; }
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 10px; overflow: hidden; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    border-radius: 8px; color: #c9d8e8; font-family: 'DM Sans', sans-serif;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700; color: #38bdf8;
    letter-spacing: -0.02em; line-height: 1.1;
}
.hero-sub { color: #6b8fad; font-size: 0.85rem; margin-top: 0.3rem; }
.section-header {
    font-family: 'Space Mono', monospace; font-size: 1rem; color: #7dd3fc;
    border-bottom: 1px solid #1e3a5f; padding-bottom: 0.4rem; margin: 1.2rem 0 0.8rem;
}
.tag { display: inline-block; background: #1e3a5f; color: #7dd3fc; border-radius: 6px;
       padding: 0.12rem 0.5rem; font-size: 0.75rem; font-family: 'Space Mono', monospace; margin: 2px; }
[data-testid="stChatMessage"] {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    border-radius: 12px; margin-bottom: 0.6rem;
}
/* Inline settings boxes */
.settings-row {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ── Sidebar — only persistent global settings, nothing tab-specific ───────────
with st.sidebar:
    st.markdown("## ⚙️ Global Settings")
    st.markdown("---")
    st.markdown("### 📊 Correlation & ML")
    start_date   = st.date_input("Data Start Date", value=pd.to_datetime("2019-01-01"))
    rolling_window= st.slider("Rolling Window (days)", 10, 90, 30)
    n_estimators = st.slider("RF Estimators", 50, 300, 100, 50)
    train_split  = st.slider("Train Split %", 60, 90, 80)
    st.markdown("---")
    st.markdown("### 🧭 Quick Navigation")
    st.markdown("""
    <div style="color:#6b8fad;font-size:0.8rem;line-height:2;">
    ☕ <b>Briefing</b> — Start here daily<br>
    📡 <b>Live Feed</b> — Real-time signals<br>
    🎯 <b>Ticker Signals</b> — Look up any ticker<br>
    💼 <b>Portfolio</b> — Holdings & P&L<br>
    📈 <b>Performance</b> — Signal track record<br>
    🔮 <b>Live Strategy</b> — Trade signals now<br>
    📊 <b>Correlations</b> — Asset relationships<br>
    🧠 <b>AI Advisor</b> — Chat with your data<br>
    🔬 <b>Backtest</b> — Strategy testing
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<span style='color:#6b8fad;font-size:0.72rem;'>FinSight AI · v1.0 · Not financial advice</span>",
                unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
col_hero, col_status = st.columns([4, 1])
with col_hero:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
      <div class="hero-title">📈 FinSight AI</div>
      <div class="hero-sub">Real-time market intelligence · AI signals · Portfolio analytics · Signal performance tracking</div>
    </div>
    """, unsafe_allow_html=True)
with col_status:
    if GROQ_API_KEY:
        st.markdown("<div style='margin-top:1.2rem;padding:0.4rem 0.8rem;background:#041a10;border:1px solid #166534;border-radius:8px;color:#4ade80;font-size:0.75rem;text-align:center;font-family:Space Mono,monospace;'>🟢 AI CONNECTED</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-top:1.2rem;padding:0.4rem 0.8rem;background:#1a0a0a;border:1px solid #7f1d1d;border-radius:8px;color:#f87171;font-size:0.75rem;text-align:center;font-family:Space Mono,monospace;'>🔴 NO API KEY</div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "☕ Briefing",
    "📡 Live Feed",
    "🎯 Ticker Signals",
    "💼 Portfolio",
    "📈 Performance",
    "🔮 Live Strategy",
    "📊 Correlations",
    "📰 Sentiment",
    "⚠️ Risk",
    "🧠 AI Advisor",
    "🔬 Backtest",
    "📜 Signal Log",
    "📧 Alerts",
])
(tab_brief, tab_live, tab_ticker, tab_port, tab_perf, tab_strat,
 tab_corr, tab_sent, tab_risk, tab_adv, tab_bt, tab_log, tab_email) = tabs

# ═══════════════════════════════════════════════════════════════════════════
# ☕ MARKET BRIEFING
# ═══════════════════════════════════════════════════════════════════════════
with tab_brief:
    st.markdown('<div class="section-header">☕ Daily Market Briefing</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Pulls overnight news from 8 sources + live prices for all major indices, sectors, and indicators.
    AI generates themes, sector outlook, top tickers to watch, key levels, and a morning playbook.
    <b style="color:#7dd3fc;">Start here every morning before the market opens.</b>
    </div>
    """, unsafe_allow_html=True)
    if not GROQ_API_KEY:
        st.error("❌ Groq API key not configured.")
    else:
        from modules.market_summary_engine import run_market_summary
        run_market_summary(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 📡 LIVE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="section-header">📡 Live Market Intelligence</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Continuously scans 10 financial news sources and generates BUY/SHORT/HOLD/WATCH signals
    with confidence, time horizon, and key risk. Every HIGH urgency signal triggers an email alert
    and is saved to your signal history automatically.
    </div>
    """, unsafe_allow_html=True)

    if not GROQ_API_KEY:
        st.error("❌ Groq API key not configured.")
    else:
        # ── Inline feed settings ───────────────────────────────────────────
        with st.expander("⚙️ Feed Settings", expanded=False):
            ls1, ls2 = st.columns(2)
            with ls1:
                refresh_interval = st.slider("Auto-refresh every (seconds)", 15, 120, 30, 5,
                                             key="live_refresh_interval")
            with ls2:
                max_age_hours = st.slider("Only show articles from last (hours)", 1, 72, 24, 1,
                                          key="live_max_age")

        if AUTOREFRESH_AVAILABLE and st.session_state.get("live_running", False):
            st_autorefresh(interval=refresh_interval * 1000, key="live_autorefresh")

        from modules.live_news_engine import run_live_news
        run_live_news(api_key=GROQ_API_KEY,
                      refresh_interval=st.session_state.get("live_refresh_interval", 30),
                      max_age_hours=st.session_state.get("live_max_age", 24))

# ═══════════════════════════════════════════════════════════════════════════
# 🎯 TICKER SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
with tab_ticker:
    st.markdown('<div class="section-header">🎯 Ticker Signal Analyser</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Type any ticker(s) → AI scans all live news sources → returns a full signal with
    bull case, bear case, individual signals with cited articles, and honest LOW confidence
    when no direct news exists. Run <b>Live Feed</b> first for the richest results.
    </div>
    """, unsafe_allow_html=True)
    if not GROQ_API_KEY:
        st.error("❌ Groq API key not configured.")
    else:
        from modules.ticker_analysis_engine import run_ticker_analysis
        run_ticker_analysis(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 💼 PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════════════════
with tab_port:
    st.markdown('<div class="section-header">💼 Portfolio Tracker</div>', unsafe_allow_html=True)
    from modules.portfolio_tracker import run_portfolio_tracker
    run_portfolio_tracker()

# ═══════════════════════════════════════════════════════════════════════════
# 📈 SIGNAL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<div class="section-header">📈 Signal Performance Tracker</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Every BUY and SHORT signal is evaluated against real price outcomes.
    Entry price is recorded at signal time. Accuracy is calculated after your chosen evaluation window.
    <b style="color:#7dd3fc;">This is your track record — the proof that the signals work.</b>
    </div>
    """, unsafe_allow_html=True)
    from modules.signal_performance import run_signal_performance
    run_signal_performance()

# ═══════════════════════════════════════════════════════════════════════════
# 📊 CORRELATION ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tab_corr:
    st.markdown('<div class="section-header">📊 Market Correlation Engine</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Enter any tickers to compute pairwise correlations, rolling correlation charts, a heatmap,
    and an AI-powered correlation predictor. Results feed into the AI Advisor and Portfolio Tracker.
    </div>
    """, unsafe_allow_html=True)
    col_in, col_hint = st.columns([3, 2])
    with col_in:
        tickers_input = st.text_input("Tickers (comma-separated)",
            placeholder="e.g. TSLA, NVDA, AMD, BTC-USD, ^GSPC")
    with col_hint:
        st.markdown('<div style="margin-top:1.5rem;"><span class="tag">STOCKS</span><span class="tag">ETFs</span><span class="tag">CRYPTO</span><span class="tag">INDICES</span></div>', unsafe_allow_html=True)
    if st.button("🚀 Run Correlation Analysis", key="run_corr"):
        if not tickers_input.strip():
            st.error("❌ Enter at least 2 tickers.")
        else:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            if len(tickers) < 2:
                st.error("❌ Need at least 2 tickers.")
            else:
                from modules.correlation_engine import run_correlation_analysis
                run_correlation_analysis(tickers=tickers, start_date=str(start_date),
                    rolling_window=rolling_window, n_estimators=n_estimators,
                    train_split=train_split/100)

# ═══════════════════════════════════════════════════════════════════════════
# 📰 NEWS SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════
with tab_sent:
    st.markdown('<div class="section-header">📰 News Sentiment Scorer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Paste any financial news URLs → AI scores each article on 5 dimensions:
    political bias, truthfulness, propaganda, hype, and panic (0–1 scale).
    Results feed into Portfolio Risk and AI Advisor.
    </div>
    """, unsafe_allow_html=True)
    urls_input = st.text_area("News article URLs (one per line)", height=140,
        placeholder="https://www.reuters.com/...\nhttps://finance.yahoo.com/...")
    if st.button("🔍 Score Articles", key="run_news"):
        if not GROQ_API_KEY:
            st.error("❌ Groq API key not configured.")
        elif not urls_input.strip():
            st.error("❌ Enter at least one URL.")
        else:
            urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
            from modules.sentiment_engine import run_sentiment_analysis
            run_sentiment_analysis(urls=urls, groq_api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# ⚠️ PORTFOLIO RISK
# ═══════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown('<div class="section-header">⚠️ Portfolio Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Combines correlation and sentiment outputs into risk flags and a composite 0–1 risk score.
    Run <b>Correlations</b> and <b>Sentiment</b> tabs first to populate this analysis.
    </div>
    """, unsafe_allow_html=True)
    from modules.portfolio_risk import run_portfolio_risk
    run_portfolio_risk()

# ═══════════════════════════════════════════════════════════════════════════
# 🧠 AI ADVISOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_adv:
    st.markdown('<div class="section-header">🧠 AI Financial Advisor</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Conversational AI grounded in your actual computed data — not generic advice.
    Reads your real correlation values, sentiment scores, and rolling trends.
    Run <b>Correlations</b> and <b>Sentiment</b> first for the richest responses.
    </div>
    """, unsafe_allow_html=True)
    if not GROQ_API_KEY:
        st.error("❌ Groq API key not configured.")
    else:
        from modules.advisor_engine import run_advisor
        run_advisor(groq_api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 🔮 LIVE STRATEGY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
with tab_strat:
    st.markdown('<div class="section-header">🔮 Live Strategy Signals</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Applies your chosen backtested strategy to <b>current market data</b> to generate a real,
    actionable trade signal right now. Shows entry price, stop loss, take profit,
    risk/reward ratio, and the exact indicator values driving the signal.
    Signals are saved to Signal History and tracked in Performance.
    </div>
    """, unsafe_allow_html=True)
    from modules.strategy_signals import run_strategy_signals
    run_strategy_signals()

# ═══════════════════════════════════════════════════════════════════════════
# 🔬 BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown('<div class="section-header">🔬 Strategy Backtester</div>', unsafe_allow_html=True)
    from modules.backtest_engine import run_backtest
    run_backtest()

# ═══════════════════════════════════════════════════════════════════════════
# 📜 SIGNAL LOG
# ═══════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown('<div class="section-header">📜 Signal History Log</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
    Every signal from every tab is automatically saved here.
    Filter, search, and download your full signal history.
    The longer this runs, the more valuable your track record becomes.
    </div>
    """, unsafe_allow_html=True)
    from modules.signal_history import run_signal_history
    run_signal_history()

# ═══════════════════════════════════════════════════════════════════════════
# 📧 EMAIL ALERTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_email:
    st.markdown('<div class="section-header">📧 Email Alert Setup</div>', unsafe_allow_html=True)
    from modules.email_alerts import render_email_setup
    render_email_setup()

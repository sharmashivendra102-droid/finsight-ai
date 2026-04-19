import streamlit as st
import pandas as pd
import sys, os
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
    page_title="FinSight AI — Financial Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
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
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 1.1rem 1.3rem; margin-bottom: 1rem;
}
.risk-flag {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d1010 100%);
    border: 1px solid #7f1d1d; border-radius: 10px;
    padding: 0.8rem 1.1rem; margin-bottom: 0.6rem; color: #fca5a5;
}
.ok-flag {
    background: linear-gradient(135deg, #041a10 0%, #062b18 100%);
    border: 1px solid #166534; border-radius: 10px;
    padding: 0.8rem 1.1rem; margin-bottom: 0.6rem; color: #86efac;
}
.metric-box {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 0.85rem; text-align: center; margin-bottom: 0.5rem;
}
.metric-box .label { font-size: 0.7rem; color: #6b8fad; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-box .value { font-size: 1.35rem; font-family: 'Space Mono', monospace; color: #38bdf8; font-weight: 700; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-weight: 700;
    letter-spacing: 0.04em; padding: 0.5rem 1.4rem; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #3b82f6);
    transform: translateY(-1px); box-shadow: 0 4px 20px rgba(14,165,233,0.35);
}
[data-baseweb="tab-list"] {
    background: #0d1b2a; border-radius: 10px;
    gap: 2px; padding: 4px; flex-wrap: wrap;
}
[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    color: #6b8fad; border-radius: 7px; padding: 0.35rem 0.7rem;
}
[aria-selected="true"][data-baseweb="tab"] { background: #1e3a5f; color: #38bdf8; }
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 10px; overflow: hidden; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    border-radius: 8px; color: #c9d8e8; font-family: 'DM Sans', sans-serif;
}
.hero-title { font-family: 'Space Mono', monospace; font-size: 1.9rem; font-weight: 700; color: #38bdf8; letter-spacing: -0.02em; }
.hero-sub { color: #6b8fad; font-size: 0.82rem; margin-top: 0.25rem; }
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.95rem; color: #7dd3fc;
    border-bottom: 1px solid #1e3a5f; padding-bottom: 0.4rem; margin: 1.1rem 0 0.8rem;
}
.tag { display: inline-block; background: #1e3a5f; color: #7dd3fc; border-radius: 6px;
       padding: 0.1rem 0.5rem; font-size: 0.72rem; font-family: 'Space Mono', monospace; margin: 2px; }
[data-testid="stChatMessage"] {
    background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 12px; margin-bottom: 0.5rem;
}
.workflow-step {
    background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 0.7rem 1rem; margin-bottom: 0.5rem; display: flex; align-items: flex-start; gap: 0.8rem;
}
.step-num {
    background: #1e3a5f; color: #38bdf8; border-radius: 50%; width: 26px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace; font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — minimal, only global ML/data settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Global Settings")
    st.caption("These affect Correlation Analytics & Backtesting")
    st.markdown("---")
    start_date    = st.date_input("Historical Data Start", value=pd.to_datetime("2019-01-01"))
    rolling_window= st.slider("Rolling Window (days)", 10, 90, 30)
    n_estimators  = st.slider("RF Tree Count", 50, 300, 100, 50)
    train_split   = st.slider("Train/Test Split %", 60, 90, 80)
    st.markdown("---")
    st.markdown("""
    <div style="color:#6b8fad;font-size:0.78rem;line-height:1.9;">
    <b style="color:#7dd3fc;">Recommended workflow:</b><br>
    ☕ → 📡 → 🎯 → 💼 daily<br>
    🔬 → 🔮 for trade signals<br>
    🧪 to test over history<br>
    📰 → 🔍 for news testing<br>
    📈 to track your record
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<span style='color:#6b8fad;font-size:0.7rem;'>FinSight AI · v1.0 · Not financial advice</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([5, 1])
with h1:
    st.markdown("""
    <div style="padding:0.8rem 0 0.4rem;">
        <div class="hero-title">📈 FinSight AI</div>
        <div class="hero-sub">Financial Intelligence Platform — Real-Time Signals · Portfolio Analytics · Strategy Testing · News Evaluation</div>
    </div>
    """, unsafe_allow_html=True)
with h2:
    badge_color = "#041a10" if GROQ_API_KEY else "#1a0a0a"
    badge_border= "#166534" if GROQ_API_KEY else "#7f1d1d"
    badge_text  = "#4ade80" if GROQ_API_KEY else "#f87171"
    badge_label = "🟢 AI ONLINE" if GROQ_API_KEY else "🔴 NO API KEY"
    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.4rem 0.7rem;background:{badge_color};
                border:1px solid {badge_border};border-radius:8px;color:{badge_text};
                font-size:0.72rem;text-align:center;font-family:'Space Mono',monospace;">
        {badge_label}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS — ordered by user journey
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "☕ Morning Briefing",        # 0  — start here daily
    "📡 Live News Feed",          # 1  — real-time signals
    "🎯 Ticker Signal Lookup",    # 2  — on-demand ticker
    "💼 Portfolio Tracker",       # 3  — your holdings
    "🔬 Strategy Backtester",     # 4  — test before you trade
    "🔮 Live Trading Signals",    # 5  — apply backtested strategy now
    "🧪 Strategy Simulator",      # 6  — historical walk-forward sim
    "📰 News Sentiment Scorer",   # 7  — score articles
    "🔍 News Signal Evaluator",   # 8  — test news accuracy
    "📈 Signal Performance",      # 9  — your real track record
    "📊 Stock Correlation Engine",# 10 — asset relationships
    "⚠️ Portfolio Risk Analysis", # 11 — risk flags
    "🧠 AI Portfolio Advisor",    # 12 — chat with your data
    "📜 Signal History Log",      # 13 — every signal saved
    "📧 Email Alert Setup",       # 14 — configure alerts
])

(tab_brief, tab_live, tab_ticker, tab_port,
 tab_bt, tab_strat, tab_sim,
 tab_sent, tab_nse, tab_perf,
 tab_corr, tab_risk, tab_adv,
 tab_log, tab_email) = tabs

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — standard tab intro card
# ─────────────────────────────────────────────────────────────────────────────
def _tab_intro(icon, title, description, tip=None):
    tip_html = f'<br><span style="color:#38bdf8;font-size:0.82rem;">💡 {tip}</span>' if tip else ""
    st.markdown(f"""
    <div class="insight-card">
        <b style="color:#7dd3fc;">{icon} {title}</b><br>
        <span style="color:#6b8fad;font-size:0.87rem;">{description}{tip_html}</span>
    </div>
    """, unsafe_allow_html=True)

def _no_key():
    st.error("❌ Groq API key not configured. Add GROQ_API_KEY to your .env file (local) or Streamlit Secrets (cloud).")

# ═══════════════════════════════════════════════════════════════════════════
# 0 — ☕ MORNING BRIEFING
# ═══════════════════════════════════════════════════════════════════════════
with tab_brief:
    st.markdown('<div class="section-header">☕ Daily Pre-Market Briefing</div>', unsafe_allow_html=True)
    _tab_intro("☕", "Start here every morning",
               "Pulls overnight news from 8 sources + live prices for all major indices, sectors, and indicators. "
               "AI generates key themes, top tickers to watch, sector outlook, key price levels, and a morning playbook.",
               "Add your watchlist in the expander below to get personalised signals every morning.")
    if not GROQ_API_KEY: _no_key()
    else:
        from modules.market_summary_engine import run_market_summary
        run_market_summary(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 1 — 📡 LIVE NEWS FEED
# ═══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="section-header">📡 Live News Intelligence Feed</div>', unsafe_allow_html=True)
    _tab_intro("📡", "Real-time news → AI stock signals",
               "Scans 10 financial news sources continuously. Every new article is analysed by AI which returns "
               "BUY / SHORT / HOLD / WATCH signals with confidence level, time horizon, affected sectors, and key risk. "
               "HIGH urgency signals are automatically saved to Signal History and trigger email alerts.",
               "Run this for 30+ minutes daily to build your news signal track record for the News Signal Evaluator.")
    if not GROQ_API_KEY: _no_key()
    else:
        with st.expander("⚙️ Feed Settings", expanded=False):
            ls1, ls2 = st.columns(2)
            with ls1:
                refresh_interval = st.slider("Auto-refresh every (seconds)", 15, 120, 30, 5, key="live_ri")
            with ls2:
                max_age_hours = st.slider("Show articles from last (hours)", 1, 72, 24, 1, key="live_mah")
        if AUTOREFRESH_AVAILABLE and st.session_state.get("live_running", False):
            st_autorefresh(interval=st.session_state.get("live_ri", 30) * 1000, key="live_ar")
        from modules.live_news_engine import run_live_news
        run_live_news(api_key=GROQ_API_KEY,
                      refresh_interval=st.session_state.get("live_ri", 30),
                      max_age_hours=st.session_state.get("live_mah", 24))

# ═══════════════════════════════════════════════════════════════════════════
# 2 — 🎯 TICKER SIGNAL LOOKUP
# ═══════════════════════════════════════════════════════════════════════════
with tab_ticker:
    st.markdown('<div class="section-header">🎯 On-Demand Ticker Signal Lookup</div>', unsafe_allow_html=True)
    _tab_intro("🎯", "Type any ticker → get a full AI signal from live news",
               "Pulls from Live News Feed session + fresh RSS feeds. Uses a large alias dictionary "
               "(e.g. XLE catches articles mentioning 'oil', 'OPEC', 'Hormuz'). Returns bull case, bear case, "
               "individual signals with cited articles, and honest LOW confidence when no direct news exists.",
               "Run the Live News Feed first — it shares its article pool with this tab for richer signals.")
    if not GROQ_API_KEY: _no_key()
    else:
        from modules.ticker_analysis_engine import run_ticker_analysis
        run_ticker_analysis(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 3 — 💼 PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════════════════
with tab_port:
    st.markdown('<div class="section-header">💼 Real-Time Portfolio Tracker</div>', unsafe_allow_html=True)
    _tab_intro("💼", "Your holdings, live prices, technicals, and news signals",
               "Enter your real positions (ticker + shares + avg cost). Shows live P&L, allocation, "
               "RSI, moving averages, beta, volatility, 1-year performance vs SPY, sector breakdown, "
               "and news signals from the Live Feed matching your holdings.",
               "Run Stock Correlation Engine with your tickers for correlation-specific risk flags in this tab.")
    from modules.portfolio_tracker import run_portfolio_tracker
    run_portfolio_tracker()

# ═══════════════════════════════════════════════════════════════════════════
# 4 — 🔬 STRATEGY BACKTESTER  (before Live Signals so user knows parameters)
# ═══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown('<div class="section-header">🔬 Strategy Backtester</div>', unsafe_allow_html=True)
    _tab_intro("🔬", "Test a strategy on historical data before using it for real trades",
               "4 academically-backed strategies: MA Crossover, RSI Mean Reversion, Momentum, and ML Weekly. "
               "Uses walk-forward validation — all results are out-of-sample only. "
               "Includes transaction costs, bootstrap Sharpe confidence intervals, and buy-and-hold benchmark.",
               "Use this tab first to find the best parameters (MA window, RSI thresholds, etc.) for your ticker. "
               "Then take those parameters to Live Trading Signals to generate today's actual signal.")
    from modules.backtest_engine import run_backtest
    run_backtest()

# ═══════════════════════════════════════════════════════════════════════════
# 5 — 🔮 LIVE TRADING SIGNALS  (after backtester)
# ═══════════════════════════════════════════════════════════════════════════
with tab_strat:
    st.markdown('<div class="section-header">🔮 Live Trading Signals</div>', unsafe_allow_html=True)
    _tab_intro("🔮", "Apply your backtested strategy to current market data",
               "Same logic as the Backtester, but runs on live prices to tell you what the strategy says to do RIGHT NOW. "
               "Shows entry price, stop loss, take profit, risk/reward ratio, and the exact indicator values "
               "driving the signal. All signals are auto-saved to Signal History.",
               "Run the Backtester first to find the best parameters for your ticker, "
               "then use those exact same parameters here.")
    from modules.strategy_signals import run_strategy_signals
    run_strategy_signals()

# ═══════════════════════════════════════════════════════════════════════════
# 6 — 🧪 STRATEGY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown('<div class="section-header">🧪 Strategy Historical Simulator</div>', unsafe_allow_html=True)
    _tab_intro("🧪", "Walk-forward simulation across multiple tickers, strategies, and time horizons",
               "Tests all 4 strategies on any tickers you choose using strict walk-forward logic "
               "(only past data at each signal point). Shows accuracy %, statistical significance (p-values), "
               "equity curves, and a ranked table of the best (strategy, ticker, horizon) combinations.",
               "Look for statistically significant combinations (p < 0.05) in the results table — "
               "those are the setups most worth using in Live Trading Signals.")
    from modules.auto_evaluator import run_auto_evaluator
    run_auto_evaluator()

# ═══════════════════════════════════════════════════════════════════════════
# 7 — 📰 NEWS SENTIMENT SCORER
# ═══════════════════════════════════════════════════════════════════════════
with tab_sent:
    st.markdown('<div class="section-header">📰 News Article Sentiment Scorer</div>', unsafe_allow_html=True)
    _tab_intro("📰", "Paste any news URL → AI scores it on 5 dimensions",
               "Scrapes the full article text, then AI scores: political bias, truthfulness, propaganda, "
               "hype, and panic on a 0–1 scale. Colour-coded results per article with averages and charts. "
               "Results feed into Portfolio Risk Analysis and the AI Portfolio Advisor.",
               "Test the scoring by pasting a known satire article (e.g. Babylon Bee) — "
               "it should score high hype and low truthfulness.")
    urls_input = st.text_area("Paste news article URLs here — one per line", height=130,
        placeholder="https://www.reuters.com/...\nhttps://www.cnbc.com/...\nhttps://finance.yahoo.com/...")
    if st.button("🔍 Score These Articles", key="run_news"):
        if not GROQ_API_KEY: _no_key()
        elif not urls_input.strip():
            st.error("❌ Paste at least one URL above.")
        else:
            urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
            from modules.sentiment_engine import run_sentiment_analysis
            run_sentiment_analysis(urls=urls, groq_api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 8 — 🔍 NEWS SIGNAL EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_nse:
    st.markdown('<div class="section-header">🔍 News Signal Accuracy Evaluator</div>', unsafe_allow_html=True)
    _tab_intro("🔍", "Did the Live Feed signals actually predict market direction?",
               "Reads every BUY and SHORT signal from your Live News Feed and Ticker Signal Lookup history, "
               "fetches the real stock price at signal time, and checks what happened 1, 3, 5, and 10 days later. "
               "Shows accuracy by confidence level, source, ticker, and time horizon with statistical significance tests.",
               "You need at least 30–50 evaluated signals for reliable conclusions. "
               "Run the Live Feed daily for 2–3 weeks, then evaluate here.")
    from modules.news_signal_evaluator import run_news_signal_evaluator
    run_news_signal_evaluator()

# ═══════════════════════════════════════════════════════════════════════════
# 9 — 📈 SIGNAL PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<div class="section-header">📈 Signal Performance Track Record</div>', unsafe_allow_html=True)
    _tab_intro("📈", "Your overall signal accuracy across ALL sources",
               "Evaluates every BUY and SHORT signal from every tab against real price outcomes. "
               "Shows overall accuracy %, accuracy by confidence level, win/loss ratio, "
               "best and worst signals, and daily accuracy over time.",
               "This is your proof that the signals work. Share this tab's results when pitching the product to users.")
    from modules.signal_performance import run_signal_performance
    run_signal_performance()

# ═══════════════════════════════════════════════════════════════════════════
# 10 — 📊 STOCK CORRELATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
with tab_corr:
    st.markdown('<div class="section-header">📊 Stock Correlation Analytics Engine</div>', unsafe_allow_html=True)
    _tab_intro("📊", "Understand how your assets move relative to each other",
               "Computes pairwise correlations, rolling correlation chart, and a full correlation heatmap "
               "for any tickers you enter. Also trains a Random Forest to predict next-step correlation. "
               "Results feed into Portfolio Risk Analysis and AI Portfolio Advisor.",
               "Enter your actual portfolio tickers to get correlation risk flags specific to your holdings.")
    col_in, col_hint = st.columns([3, 2])
    with col_in:
        tickers_input = st.text_input("Enter tickers separated by commas",
            placeholder="e.g. TSLA, NVDA, AMD, BTC-USD, ^GSPC, SPY")
    with col_hint:
        st.markdown('<div style="margin-top:1.4rem;"><span class="tag">STOCKS</span><span class="tag">ETFs</span><span class="tag">CRYPTO</span><span class="tag">INDICES</span></div>', unsafe_allow_html=True)
    if st.button("🚀 Run Correlation Analysis", key="run_corr"):
        if not tickers_input.strip():
            st.error("❌ Enter at least 2 tickers separated by commas.")
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
# 11 — ⚠️ PORTFOLIO RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown('<div class="section-header">⚠️ Portfolio Risk Analysis</div>', unsafe_allow_html=True)
    _tab_intro("⚠️", "Combines correlation and sentiment data into a risk score",
               "Automatically reads results from Stock Correlation Engine and News Sentiment Scorer. "
               "Flags high correlation clusters, natural hedges, high hype + low truthfulness, "
               "and generates a composite 0–1 risk score (LOW / MODERATE / HIGH).",
               "Run Stock Correlation Engine and News Sentiment Scorer first for the full analysis.")
    from modules.portfolio_risk import run_portfolio_risk
    run_portfolio_risk()

# ═══════════════════════════════════════════════════════════════════════════
# 12 — 🧠 AI PORTFOLIO ADVISOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_adv:
    st.markdown('<div class="section-header">🧠 AI Portfolio Advisor</div>', unsafe_allow_html=True)
    _tab_intro("🧠", "Conversational AI grounded in your actual computed data",
               "Reads your real correlation values, sentiment scores, and rolling trends as context — "
               "not generic financial knowledge. Gives structured advice with sections for correlation insights, "
               "sentiment outlook, risk assessment, and specific recommendations.",
               "Run Stock Correlation Engine and News Sentiment Scorer first — the more data, the richer the advice.")
    if not GROQ_API_KEY: _no_key()
    else:
        from modules.advisor_engine import run_advisor
        run_advisor(groq_api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 13 — 📜 SIGNAL HISTORY LOG
# ═══════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown('<div class="section-header">📜 Signal History Log</div>', unsafe_allow_html=True)
    _tab_intro("📜", "Every signal from every tab, saved automatically",
               "All BUY/SHORT/HOLD/WATCH signals from Live News Feed, Ticker Signal Lookup, "
               "Live Trading Signals, and Market Briefing are saved to a local database. "
               "Filter, search, and download your full history.")
    from modules.signal_history import run_signal_history
    run_signal_history()

# ═══════════════════════════════════════════════════════════════════════════
# 14 — 📧 EMAIL ALERT SETUP
# ═══════════════════════════════════════════════════════════════════════════
with tab_email:
    st.markdown('<div class="section-header">📧 Email Alert Configuration</div>', unsafe_allow_html=True)
    _tab_intro("📧", "Get emailed automatically when a HIGH urgency signal is detected",
               "Configure Gmail or SendGrid. Once set up, any HIGH urgency signal from the Live News Feed "
               "will send a formatted email with the article, market impact, and all stock recommendations.")
    from modules.email_alerts import render_email_setup
    render_email_setup()



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
    page_title="FinSight AI — Financial Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.risk-flag {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d1010 100%);
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
    color: #fca5a5;
}
.ok-flag {
    background: linear-gradient(135deg, #041a10 0%, #062b18 100%);
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
    color: #86efac;
}
.metric-box {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-box .label { font-size: 0.78rem; color: #6b8fad; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-box .value { font-size: 1.6rem; font-family: 'Space Mono', monospace; color: #38bdf8; font-weight: 700; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-weight: 700;
    letter-spacing: 0.05em; padding: 0.6rem 2rem; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #3b82f6);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(14,165,233,0.35);
}
[data-baseweb="tab-list"] { background: #0d1b2a; border-radius: 10px; gap: 4px; padding: 4px; }
[data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 0.82rem; color: #6b8fad; border-radius: 8px; }
[aria-selected="true"][data-baseweb="tab"] { background: #1e3a5f; color: #38bdf8; }
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 10px; overflow: hidden; }
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    border-radius: 8px; color: #c9d8e8; font-family: 'DM Sans', sans-serif;
}
.hero-title { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #38bdf8; letter-spacing: -0.02em; line-height: 1.1; }
.hero-sub { color: #6b8fad; font-size: 1rem; margin-top: 0.4rem; }
.section-header {
    font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #7dd3fc;
    border-bottom: 1px solid #1e3a5f; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem;
}
.tag { display: inline-block; background: #1e3a5f; color: #7dd3fc; border-radius: 6px; padding: 0.15rem 0.6rem; font-size: 0.78rem; font-family: 'Space Mono', monospace; margin: 2px; }
[data-testid="stChatMessage"] {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh for live news tab ────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    st.markdown("### 📅 Data Settings")
    start_date     = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
    rolling_window = st.slider("Rolling Window (days)", 10, 90, 30)
    st.markdown("---")
    st.markdown("### 🤖 ML Settings")
    n_estimators = st.slider("RF Estimators", 50, 300, 100, 50)
    train_split  = st.slider("Train Split %", 60, 90, 80)
    st.markdown("---")
    st.markdown("### 📡 Live Feed Settings")
    refresh_interval = st.slider("Refresh interval (seconds)", 15, 120, 30, 5)
    st.markdown("---")
    st.markdown("### 🧠 Recommended Workflow")
    st.markdown("""
    <div style="color:#6b8fad;font-size:0.82rem;line-height:1.8;">
    1️⃣ Enter tickers → Run Correlation<br>
    2️⃣ Paste news URLs → Run Sentiment<br>
    3️⃣ Check Portfolio Risk<br>
    4️⃣ Chat with AI Advisor<br>
    5️⃣ Start Live News Feed
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<span style='color:#6b8fad;font-size:0.78rem;'>FinSight AI · MVP v1.0<br>Built for retail investors & traders</span>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem;">
  <div class="hero-title">📈 FinSight AI</div>
  <div class="hero-sub">Financial Analytics · Sentiment · Portfolio Risk · AI Advisor · Live News Intelligence</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Correlation Analytics",
    "📰  News Sentiment",
    "⚠️  Portfolio Risk",
    "🧠  AI Advisor",
    "📡  Live Intelligence",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CORRELATION ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 Market Correlation Engine</div>', unsafe_allow_html=True)
    col_in, col_hint = st.columns([3, 2])
    with col_in:
        tickers_input = st.text_input(
            "Enter Tickers (comma-separated)",
            placeholder="e.g. TSLA, NVDA, AMD, BTC-USD, ^GSPC",
        )
    with col_hint:
        st.markdown('<div class="insight-card" style="margin-top:1.6rem;padding:0.8rem 1.2rem;"><span class="tag">STOCKS</span><span class="tag">ETFs</span><span class="tag">CRYPTO</span><span class="tag">INDICES</span></div>', unsafe_allow_html=True)
    if st.button("🚀 Run Correlation Analysis", key="run_corr"):
        if not tickers_input.strip():
            st.error("❌ Please enter at least 2 tickers.")
        else:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            if len(tickers) < 2:
                st.error("❌ Need at least 2 tickers.")
            else:
                from modules.correlation_engine import run_correlation_analysis
                run_correlation_analysis(tickers=tickers, start_date=str(start_date),
                    rolling_window=rolling_window, n_estimators=n_estimators, train_split=train_split/100)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — NEWS SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📰 AI News Sentiment Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-card"><b style="color:#7dd3fc;">How it works:</b><br><span style="color:#6b8fad;font-size:0.9rem;">Paste financial news URLs → articles are scraped → AI scores each article across 5 dimensions. Results feed into the AI Advisor.</span></div>', unsafe_allow_html=True)
    urls_input = st.text_area("Enter News URLs (one per line)", height=180,
        placeholder="https://www.reuters.com/...\nhttps://finance.yahoo.com/...")
    if st.button("🔍 Run News Analysis", key="run_news"):
        if not GROQ_API_KEY:
            st.error("❌ Groq API key not configured.")
        elif not urls_input.strip():
            st.error("❌ Please enter at least one URL.")
        else:
            urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
            from modules.sentiment_engine import run_sentiment_analysis
            run_sentiment_analysis(urls=urls, groq_api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO RISK
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">⚠️ Portfolio Risk Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-card"><b style="color:#7dd3fc;">Risk Signal Logic:</b><br><span style="color:#6b8fad;font-size:0.9rem;">Combines correlation clusters and sentiment scores to surface portfolio risk flags.</span></div>', unsafe_allow_html=True)
    from modules.portfolio_risk import run_portfolio_risk
    run_portfolio_risk()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — AI ADVISOR
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🧠 AI Financial Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-card"><b style="color:#7dd3fc;">How it works:</b><br><span style="color:#6b8fad;font-size:0.9rem;">Reads your actual correlation data, sentiment scores, and risk signals — then gives specific, grounded financial advice. Run the other tabs first for the richest advice.</span></div>', unsafe_allow_html=True)
    if not GROQ_API_KEY:
        st.error("❌ Groq API key not configured.")
    else:
        from modules.advisor_engine import run_advisor
        run_advisor(groq_api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📡 Live Market Intelligence</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Real-time news → AI stock signals</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Scans {len(__import__('modules.live_news_engine', fromlist=['RSS_FEEDS']).RSS_FEEDS)} live financial news sources every {refresh_interval} seconds.
        New articles are instantly analysed by AI which returns BUY / SHORT / HOLD signals
        with reasoning, confidence, and time horizon. No manual input needed — just press Start.
        </span>
    </div>
    """, unsafe_allow_html=True)

    if not GROQ_API_KEY:
        st.error("❌ Groq API key not configured.")
    else:
        # Auto-refresh only fires when live feed is running
        if AUTOREFRESH_AVAILABLE and st.session_state.get("live_running", False):
            st_autorefresh(interval=refresh_interval * 1000, key="live_refresh")

        from modules.live_news_engine import run_live_news
        run_live_news(api_key=GROQ_API_KEY, refresh_interval=refresh_interval)

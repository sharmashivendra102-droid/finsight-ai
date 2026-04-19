import streamlit as st
import pandas as pd
import sys, os
from pathlib import Path
from datetime import datetime

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
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Instrument+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg-void:     #04080f;
  --bg-base:     #070d18;
  --bg-surface:  #0b1525;
  --bg-elevated: #0f1e35;
  --bg-hover:    #152540;

  --border-subtle: rgba(56,189,248,.07);
  --border-default: rgba(56,189,248,.14);
  --border-strong: rgba(56,189,248,.28);

  --accent-cyan:   #38bdf8;
  --accent-cyan-dim: rgba(56,189,248,.12);
  --accent-green:  #34d399;
  --accent-green-dim: rgba(52,211,153,.12);
  --accent-amber:  #fbbf24;
  --accent-amber-dim: rgba(251,191,36,.12);
  --accent-red:    #f87171;
  --accent-red-dim: rgba(248,113,113,.12);
  --accent-violet: #a78bfa;

  --text-primary:  #e2eaf4;
  --text-secondary: #8ba3c1;
  --text-muted:    #4d6a89;
  --text-accent:   #38bdf8;

  --font-display: 'Syne', sans-serif;
  --font-body:    'Instrument Sans', sans-serif;
  --font-mono:    'IBM Plex Mono', monospace;

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-xl: 22px;

  --shadow-glow: 0 0 40px rgba(56,189,248,.08);
  --shadow-card: 0 4px 24px rgba(0,0,0,.4);
}

/* ── Reset & base ── */
html, body, [class*="css"] {
  font-family: var(--font-body);
  background: var(--bg-void) !important;
  color: var(--text-primary);
}

.stApp {
  background: var(--bg-void) !important;
}

/* Subtle dot-grid background */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: radial-gradient(circle, rgba(56,189,248,.04) 1px, transparent 1px);
  background-size: 32px 32px;
  pointer-events: none;
  z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border-default) !important;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] .stSlider { margin-bottom: 1rem; }

/* ── Tab bar ── */
[data-baseweb="tab-list"] {
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-subtle);
  border-radius: 0;
  gap: 0;
  padding: 0 4px;
  overflow-x: auto;
}
[data-baseweb="tab"] {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  font-weight: 500;
  letter-spacing: .04em;
  text-transform: uppercase;
  color: var(--text-muted);
  border-radius: 0;
  padding: 0.75rem 1rem;
  border-bottom: 2px solid transparent;
  transition: all .15s;
  white-space: nowrap;
}
[data-baseweb="tab"]:hover { color: var(--text-secondary); }
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--accent-cyan) !important;
  border-bottom-color: var(--accent-cyan);
  background: transparent;
}

/* ── Buttons ── */
.stButton > button {
  background: transparent;
  border: 1px solid var(--border-default);
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 500;
  letter-spacing: .06em;
  text-transform: uppercase;
  border-radius: var(--radius-sm);
  padding: .5rem 1.2rem;
  transition: all .15s;
}
.stButton > button:hover {
  border-color: var(--accent-cyan);
  color: var(--accent-cyan);
  background: var(--accent-cyan-dim);
  box-shadow: 0 0 20px rgba(56,189,248,.15);
}

/* Primary button variant */
.btn-primary > button {
  background: var(--accent-cyan) !important;
  border-color: var(--accent-cyan) !important;
  color: var(--bg-void) !important;
  font-weight: 700 !important;
}
.btn-primary > button:hover {
  background: #7dd3fc !important;
  box-shadow: 0 0 30px rgba(56,189,248,.4) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.875rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent-cyan) !important;
  box-shadow: 0 0 0 3px var(--accent-cyan-dim) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  margin-bottom: .5rem;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  background: var(--bg-surface) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent-cyan) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 2px; }

/* ═══════════════════════════════
   COMPONENT CLASSES
═══════════════════════════════ */

/* Card */
.card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.4rem;
  margin-bottom: .75rem;
  position: relative;
  overflow: hidden;
  transition: border-color .2s;
}
.card:hover { border-color: var(--border-default); }
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(56,189,248,.2), transparent);
}

.card-accent-cyan  { border-color: rgba(56,189,248,.2); }
.card-accent-green { border-color: rgba(52,211,153,.2); }
.card-accent-amber { border-color: rgba(251,191,36,.2);  }
.card-accent-red   { border-color: rgba(248,113,113,.2); }

/* Metric box */
.metric {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: .9rem 1rem;
  text-align: center;
  margin-bottom: .5rem;
  transition: all .2s;
}
.metric:hover { border-color: var(--border-default); }
.metric .m-label {
  font-family: var(--font-mono);
  font-size: .62rem;
  font-weight: 500;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: .35rem;
}
.metric .m-value {
  font-family: var(--font-mono);
  font-size: 1.4rem;
  font-weight: 600;
  color: var(--accent-cyan);
  line-height: 1;
}
.metric .m-delta {
  font-family: var(--font-mono);
  font-size: .7rem;
  color: var(--text-muted);
  margin-top: .25rem;
}

/* Section header */
.section-header {
  font-family: var(--font-mono);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border-subtle);
  padding-bottom: .5rem;
  margin: 1.4rem 0 .9rem;
}

/* Status dot */
.dot-live { width:7px;height:7px;border-radius:50%;background:var(--accent-green);
  display:inline-block;box-shadow:0 0 6px var(--accent-green);animation:pulse 2s infinite; }
.dot-idle { width:7px;height:7px;border-radius:50%;background:var(--text-muted);display:inline-block; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.4} }

/* Badge */
.badge {
  display: inline-flex;
  align-items: center;
  gap: .3rem;
  font-family: var(--font-mono);
  font-size: .65rem;
  font-weight: 600;
  letter-spacing: .06em;
  padding: .2rem .6rem;
  border-radius: 99px;
  border: 1px solid;
}
.badge-buy   { color:var(--accent-green);border-color:rgba(52,211,153,.3);background:var(--accent-green-dim); }
.badge-short { color:var(--accent-red);  border-color:rgba(248,113,113,.3);background:var(--accent-red-dim); }
.badge-hold  { color:var(--accent-amber);border-color:rgba(251,191,36,.3); background:var(--accent-amber-dim); }
.badge-watch { color:var(--accent-cyan); border-color:rgba(56,189,248,.3); background:var(--accent-cyan-dim); }
.badge-high  { color:var(--accent-cyan); border-color:rgba(56,189,248,.3); background:var(--accent-cyan-dim); }
.badge-med   { color:var(--accent-amber);border-color:rgba(251,191,36,.3); background:var(--accent-amber-dim); }
.badge-low   { color:var(--text-muted);  border-color:var(--border-subtle);background:var(--bg-elevated); }

/* Command tile */
.cmd-tile {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 1.2rem 1.3rem;
  cursor: pointer;
  transition: all .2s;
  height: 100%;
  position: relative;
  overflow: hidden;
}
.cmd-tile:hover {
  border-color: var(--border-strong);
  background: var(--bg-elevated);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,0,0,.3), var(--shadow-glow);
}
.cmd-tile .ct-icon { font-size: 1.6rem; margin-bottom: .6rem; }
.cmd-tile .ct-title {
  font-family: var(--font-display);
  font-size: .9rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: .3rem;
}
.cmd-tile .ct-desc {
  font-size: .75rem;
  color: var(--text-muted);
  line-height: 1.5;
}
.cmd-tile .ct-tag {
  position: absolute;
  top: .8rem; right: .8rem;
  font-family: var(--font-mono);
  font-size: .55rem;
  font-weight: 700;
  letter-spacing: .08em;
  padding: .15rem .5rem;
  border-radius: 99px;
  text-transform: uppercase;
}
.ct-tag-daily { background:rgba(52,211,153,.12);color:var(--accent-green);border:1px solid rgba(52,211,153,.2); }
.ct-tag-pro   { background:rgba(56,189,248,.10);color:var(--accent-cyan); border:1px solid rgba(56,189,248,.2); }
.ct-tag-test  { background:rgba(251,191,36,.10);color:var(--accent-amber);border:1px solid rgba(251,191,36,.2); }

/* Tag chip */
.tag {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: .62rem;
  font-weight: 500;
  padding: .15rem .5rem;
  border-radius: 4px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  letter-spacing: .04em;
  margin: 1px;
}

/* Risk flags */
.flag-red {
  background: rgba(248,113,113,.06);
  border: 1px solid rgba(248,113,113,.2);
  border-radius: var(--radius-md);
  padding: .8rem 1rem;
  margin-bottom: .5rem;
  color: #fca5a5;
}
.flag-amber {
  background: rgba(251,191,36,.06);
  border: 1px solid rgba(251,191,36,.2);
  border-radius: var(--radius-md);
  padding: .8rem 1rem;
  margin-bottom: .5rem;
  color: #fde68a;
}
.flag-green {
  background: rgba(52,211,153,.06);
  border: 1px solid rgba(52,211,153,.2);
  border-radius: var(--radius-md);
  padding: .8rem 1rem;
  margin-bottom: .5rem;
  color: #6ee7b7;
}

/* Insight card */
.insight-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 1.1rem 1.3rem;
  margin-bottom: .75rem;
}

/* OK / risk flags (used by portfolio_risk) */
.risk-flag { background:rgba(248,113,113,.06);border:1px solid rgba(248,113,113,.2);border-radius:var(--radius-md);padding:.8rem 1rem;margin-bottom:.5rem;color:#fca5a5; }
.ok-flag   { background:rgba(52,211,153,.06); border:1px solid rgba(52,211,153,.2); border-radius:var(--radius-md);padding:.8rem 1rem;margin-bottom:.5rem;color:#6ee7b7; }

/* ═══ HERO ═══ */
.hero {
  padding: 2.5rem 0 1.5rem;
  position: relative;
}
.hero-eyebrow {
  font-family: var(--font-mono);
  font-size: .65rem;
  font-weight: 600;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  margin-bottom: .6rem;
}
.hero-title {
  font-family: var(--font-display);
  font-size: 2.6rem;
  font-weight: 800;
  color: var(--text-primary);
  letter-spacing: -.02em;
  line-height: 1;
  margin-bottom: .5rem;
}
.hero-title span { color: var(--accent-cyan); }
.hero-sub {
  font-size: .88rem;
  color: var(--text-muted);
  max-width: 560px;
  line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — only ML/data globals
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:var(--font-mono);font-size:.65rem;letter-spacing:.15em;
                text-transform:uppercase;color:var(--text-muted);padding:.5rem 0 1rem;">
        ⚙ Global Config
    </div>""", unsafe_allow_html=True)

    start_date    = st.date_input("Data Start", value=pd.to_datetime("2019-01-01"))
    rolling_window= st.slider("Rolling Window (days)", 10, 90, 30)
    n_estimators  = st.slider("RF Trees", 50, 300, 100, 50)
    train_split   = st.slider("Train Split %", 60, 90, 80)

    st.markdown("---")
    ai_status = "🟢  AI Online" if GROQ_API_KEY else "🔴  No API Key"
    ai_color  = "#34d399" if GROQ_API_KEY else "#f87171"
    st.markdown(f"""
    <div style="font-family:var(--font-mono);font-size:.72rem;color:{ai_color};">
        {ai_status}
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:var(--font-mono);font-size:.6rem;color:var(--text-muted);
                margin-top:1rem;line-height:2;">
        FinSight AI · v1.0<br>Not financial advice
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([5, 1])
with h1:
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Financial Intelligence Platform</div>
      <div class="hero-title">Fin<span>Sight</span> AI</div>
      <div class="hero-sub">
        Real-time market signals · Portfolio analytics · Strategy testing · Self-improving intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)
with h2:
    now_str = datetime.now().strftime("%H:%M")
    date_str= datetime.now().strftime("%a %b %d")
    st.markdown(f"""
    <div style="text-align:right;padding-top:2rem;">
      <div style="font-family:var(--font-mono);font-size:1.4rem;font-weight:600;
                  color:var(--text-primary);">{now_str}</div>
      <div style="font-family:var(--font-mono);font-size:.7rem;color:var(--text-muted);
                  letter-spacing:.08em;margin-top:.2rem;">{date_str}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
TABS = [
    "⌂  Command Center",
    "☕  Morning Briefing",
    "📡  Live Feed",
    "🎯  Ticker Signals",
    "💼  Portfolio",
    "🔬  Backtester",
    "🔮  Live Signals",
    "🧪  Simulator",
    "📰  Sentiment",
    "🔍  News Accuracy",
    "📈  Performance",
    "🧠  Profitability",
    "📊  Correlations",
    "⚠   Risk",
    "🤖  AI Advisor",
    "📜  Signal Log",
    "📧  Alerts",
]

tabs = st.tabs(TABS)
(tab_home, tab_brief, tab_live, tab_ticker, tab_port,
 tab_bt, tab_strat, tab_sim,
 tab_sent, tab_nse, tab_perf, tab_spe,
 tab_corr, tab_risk, tab_adv,
 tab_log, tab_email) = tabs


# ─── helpers ─────────────────────────────────────────────────────────────────
def _no_key():
    st.markdown("""
    <div class="card card-accent-red" style="text-align:center;padding:2rem;">
      <div style="font-size:1.5rem;margin-bottom:.5rem;">🔴</div>
      <div style="font-family:var(--font-mono);font-size:.75rem;color:var(--accent-red);
                  font-weight:600;letter-spacing:.06em;">GROQ API KEY NOT CONFIGURED</div>
      <div style="color:var(--text-muted);font-size:.8rem;margin-top:.5rem;">
        Add GROQ_API_KEY to your .env file or Streamlit Secrets
      </div>
    </div>""", unsafe_allow_html=True)


def _tab_card(icon, title, desc, tip=None):
    tip_html = f'<div style="margin-top:.6rem;padding:.5rem .8rem;background:var(--bg-elevated);border-radius:var(--radius-sm);border-left:2px solid var(--accent-cyan);font-size:.78rem;color:var(--text-secondary);">💡 {tip}</div>' if tip else ""
    st.markdown(f"""
    <div class="card">
      <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem;">
        <span style="font-size:1.1rem;">{icon}</span>
        <span style="font-family:var(--font-display);font-weight:700;
                     font-size:.95rem;color:var(--text-primary);">{title}</span>
      </div>
      <div style="font-size:.82rem;color:var(--text-muted);line-height:1.6;">{desc}</div>
      {tip_html}
    </div>""", unsafe_allow_html=True)


def _metric(label, value, color=None, delta=None):
    c = color or "var(--accent-cyan)"
    d = f'<div class="m-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric">
      <div class="m-label">{label}</div>
      <div class="m-value" style="color:{c};">{value}</div>
      {d}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 0 — ⌂ COMMAND CENTER
# ═══════════════════════════════════════════════════════════════════════════
with tab_home:

    # ── Status strip ──────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    live_running = st.session_state.get("live_running", False)
    sig_count    = len(st.session_state.get("processed_signals", []))
    port_count   = 0
    try:
        import sqlite3
        conn = sqlite3.connect(str(APP_DIR / "portfolio.db"), check_same_thread=False)
        port_count = conn.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]
        conn.close()
    except Exception:
        pass
    sig_hist = 0
    try:
        conn2 = sqlite3.connect(str(APP_DIR / "signal_history.db"), check_same_thread=False)
        sig_hist = conn2.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        conn2.close()
    except Exception:
        pass

    with s1:
        dot = '<span class="dot-live"></span>' if live_running else '<span class="dot-idle"></span>'
        status = "LIVE" if live_running else "IDLE"
        st.markdown(f"""
        <div class="metric">
          <div class="m-label">Live Feed</div>
          <div style="display:flex;align-items:center;justify-content:center;gap:.4rem;margin:.3rem 0;">
            {dot}
            <span style="font-family:var(--font-mono);font-size:.85rem;font-weight:600;
                         color:{'var(--accent-green)' if live_running else 'var(--text-muted)'};">{status}</span>
          </div>
          <div class="m-delta">{sig_count} signals this session</div>
        </div>""", unsafe_allow_html=True)

    with s2:
        _metric("Signal History", f"{sig_hist:,}", delta="total logged signals")
    with s3:
        _metric("Portfolio", f"{port_count}", "var(--accent-violet)", "active positions")
    with s4:
        ai_c = "var(--accent-green)" if GROQ_API_KEY else "var(--accent-red)"
        ai_v = "CONNECTED" if GROQ_API_KEY else "OFFLINE"
        _metric("AI Engine", ai_v, ai_c, "llama-3.3-70b")

    st.markdown('<div class="section-header">Daily Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.8rem;color:var(--text-muted);margin-bottom:1rem;">
      Follow this sequence every morning for the best results.
    </div>""", unsafe_allow_html=True)

    # ── Workflow tiles row 1 ───────────────────────────────────────────────
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("""
        <div class="cmd-tile">
          <span class="ct-tag ct-tag-daily">Daily</span>
          <div class="ct-icon">☕</div>
          <div class="ct-title">Morning Briefing</div>
          <div class="ct-desc">Start here. AI-generated pre-market summary with live prices, sector outlook, and top tickers to watch.</div>
        </div>""", unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="cmd-tile">
          <span class="ct-tag ct-tag-daily">Daily</span>
          <div class="ct-icon">📡</div>
          <div class="ct-title">Live News Feed</div>
          <div class="ct-desc">Press Start. Scans 10 sources continuously. AI generates BUY/SHORT signals as news breaks.</div>
        </div>""", unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="cmd-tile">
          <span class="ct-tag ct-tag-daily">Daily</span>
          <div class="ct-icon">🎯</div>
          <div class="ct-title">Ticker Signals</div>
          <div class="ct-desc">Type any ticker. AI scans live news and returns a full signal with bull/bear case and cited sources.</div>
        </div>""", unsafe_allow_html=True)
    with w4:
        st.markdown("""
        <div class="cmd-tile">
          <span class="ct-tag ct-tag-daily">Daily</span>
          <div class="ct-icon">💼</div>
          <div class="ct-title">Portfolio Tracker</div>
          <div class="ct-desc">Live P&L, RSI, moving averages, beta, volatility, and news signals for your holdings.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:.75rem;'></div>", unsafe_allow_html=True)

    # ── Workflow tiles row 2 ───────────────────────────────────────────────
    w5, w6, w7, w8 = st.columns(4)
    with w5:
        st.markdown("""
        <div class="cmd-tile">
          <span class="ct-tag ct-tag-test">Test First</span>
          <div class="ct-icon">🔬</div>
          <div class="ct-title">Backtester</div>
          <div class="ct-desc">Test MA, RSI, Momentum, or ML strategies on history. Walk-forward OOS only. Find your best parameters here.</div>
        </div>""", unsafe_allow_html=True)
    with w6:
        st.markdown("""
        <div class="cmd-tile">
          <span class="ct-tag ct-tag-pro">Then Trade</span>
          <div class="ct-icon">🔮</div>
          <div class="ct-title">Live Signals</div>
          <div class="ct-desc">Apply your backtested strategy to current prices. Entry, stop loss, take profit, and R/R ratio — right now.</div>
        </div>""", unsafe_allow_html=True)
    with w7:
        st.markdown("""
        <div class="cmd-tile">
          <span class="ct-tag ct-tag-pro">Analyse</span>
          <div class="ct-icon">📈</div>
          <div class="ct-title">Performance</div>
          <div class="ct-desc">Every signal evaluated against real prices. Horizon-matched, reversal-aware, external shocks excluded.</div>
        </div>""", unsafe_allow_html=True)
    with w8:
        st.markdown("""
        <div class="cmd-tile">
          <span class="ct-tag ct-tag-pro">Optimise</span>
          <div class="ct-icon">🧠</div>
          <div class="ct-title">Profitability Engine</div>
          <div class="ct-desc">Which signal types make money? In which market regimes? Auto-ranks strategies. Kill bad ones.</div>
        </div>""", unsafe_allow_html=True)

    # ── Feature reference grid ─────────────────────────────────────────────
    st.markdown('<div class="section-header">All Features</div>', unsafe_allow_html=True)

    features = [
        ("🧪", "Strategy Simulator",     "Walk-forward historical simulation across any tickers. p-values, equity curves, best combo finder.",           "Test"),
        ("📰", "News Sentiment",         "Paste any article URL. AI scores political bias, truthfulness, propaganda, hype, and panic (0–1).",             "Analysis"),
        ("🔍", "News Accuracy",          "Did your Live Feed signals predict direction? Evaluates every signal against real prices by confidence level.",  "Evaluation"),
        ("📊", "Correlations",           "Pairwise correlation table, heatmap, rolling chart, and RF predictor for any tickers.",                         "Analysis"),
        ("⚠",  "Portfolio Risk",        "Combines correlation and sentiment into a composite 0–1 risk score with specific flags.",                        "Risk"),
        ("🤖", "AI Advisor",            "Conversational AI grounded in your real correlation and sentiment data — not generic advice.",                   "AI"),
        ("📜", "Signal History",         "Every signal from every tab saved automatically to SQLite. Filterable, searchable, downloadable.",               "Data"),
        ("📧", "Email Alerts",           "HIGH urgency Live Feed signals automatically trigger email alerts via Gmail or SendGrid.",                       "Alerts"),
    ]

    feat_cols = st.columns(4)
    for i, (icon, name, desc, tag) in enumerate(features):
        tag_color = {"Test":"var(--accent-amber)","Analysis":"var(--accent-cyan)",
                     "Evaluation":"var(--accent-violet)","Risk":"var(--accent-red)",
                     "AI":"var(--accent-cyan)","Data":"var(--text-muted)","Alerts":"var(--accent-green)"}.get(tag,"var(--text-muted)")
        with feat_cols[i % 4]:
            st.markdown(f"""
            <div class="card" style="margin-bottom:.6rem;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.4rem;">
                <span style="font-size:1rem;">{icon}</span>
                <span style="font-family:var(--font-mono);font-size:.55rem;font-weight:700;
                             letter-spacing:.06em;color:{tag_color};text-transform:uppercase;">{tag}</span>
              </div>
              <div style="font-family:var(--font-display);font-weight:700;font-size:.82rem;
                          color:var(--text-primary);margin-bottom:.3rem;">{name}</div>
              <div style="font-size:.75rem;color:var(--text-muted);line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Quick start ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Quick Start Guide</div>', unsafe_allow_html=True)
    steps = [
        ("1", "var(--accent-cyan)",   "Run Morning Briefing",      "Opens with live prices and AI-generated market themes for today."),
        ("2", "var(--accent-cyan)",   "Start Live News Feed",      "Press ▶️ Start. Leave it running. Signals auto-log to your history."),
        ("3", "var(--accent-amber)",  "Backtest a strategy",       "Try MA Crossover on QQQ from 2018. Note the best parameters."),
        ("4", "var(--accent-amber)",  "Generate a Live Signal",    "Use those same parameters in Live Signals for today's actual trade setup."),
        ("5", "var(--accent-green)",  "Check Signal Performance",  "After 5+ trading days, evaluate how your signals did against real prices."),
        ("6", "var(--accent-green)",  "Review Profitability Engine","After 2–3 weeks, see which signal types actually make money."),
    ]
    qs_cols = st.columns(3)
    for i, (num, color, title, desc) in enumerate(steps):
        with qs_cols[i % 3]:
            st.markdown(f"""
            <div class="card" style="margin-bottom:.6rem;">
              <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;">
                <div style="width:22px;height:22px;border-radius:50%;background:{color};
                            display:flex;align-items:center;justify-content:center;
                            font-family:var(--font-mono);font-size:.65rem;font-weight:700;
                            color:var(--bg-void);flex-shrink:0;">{num}</div>
                <div style="font-family:var(--font-display);font-weight:700;font-size:.82rem;
                            color:var(--text-primary);">{title}</div>
              </div>
              <div style="font-size:.75rem;color:var(--text-muted);line-height:1.5;
                          padding-left:1.8rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1 — ☕ MORNING BRIEFING
# ═══════════════════════════════════════════════════════════════════════════
with tab_brief:
    st.markdown('<div class="section-header">Morning Briefing</div>', unsafe_allow_html=True)
    _tab_card("☕", "Daily Pre-Market Summary",
              "Overnight news from 8 sources + live prices for all major indices, sectors, and indicators. "
              "AI generates themes, sector outlook, top tickers to watch, key levels, and a morning playbook.",
              "Add your watchlist in the expander below for personalised signals every morning.")
    if not GROQ_API_KEY: _no_key()
    else:
        from modules.market_summary_engine import run_market_summary
        run_market_summary(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 2 — 📡 LIVE NEWS FEED
# ═══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown('<div class="section-header">Live News Intelligence Feed</div>', unsafe_allow_html=True)
    _tab_card("📡", "Real-time news → AI stock signals",
              "Scans 10 financial news sources. Every new article is analysed — returns BUY/SHORT/HOLD/WATCH "
              "with confidence, time horizon, and key risk. HIGH urgency signals email you automatically and log to history.",
              "Run for 30+ minutes daily to build your signal track record for the News Accuracy tab.")
    if not GROQ_API_KEY: _no_key()
    else:
        with st.expander("⚙ Feed Settings", expanded=False):
            ls1, ls2 = st.columns(2)
            with ls1: refresh_interval = st.slider("Auto-refresh (seconds)", 15, 120, 30, 5, key="live_ri")
            with ls2: max_age_hours    = st.slider("Article age limit (hours)", 1, 72, 24, 1, key="live_mah")
        if AUTOREFRESH_AVAILABLE and st.session_state.get("live_running", False):
            st_autorefresh(interval=st.session_state.get("live_ri", 30) * 1000, key="live_ar")
        from modules.live_news_engine import run_live_news
        run_live_news(api_key=GROQ_API_KEY,
                      refresh_interval=st.session_state.get("live_ri", 30),
                      max_age_hours=st.session_state.get("live_mah", 24))

# ═══════════════════════════════════════════════════════════════════════════
# 3 — 🎯 TICKER SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
with tab_ticker:
    st.markdown('<div class="section-header">On-Demand Ticker Signal Lookup</div>', unsafe_allow_html=True)
    _tab_card("🎯", "Type any ticker → full AI signal from live news",
              "Pulls from Live Feed session + fresh RSS. Large alias dictionary catches company names, "
              "exec names, and sector keywords. Returns bull case, bear case, individual signals with cited articles. "
              "Honest LOW confidence when no direct news exists.",
              "Run Live Feed first — it shares its article pool with this tab for richer signals.")
    if not GROQ_API_KEY: _no_key()
    else:
        from modules.ticker_analysis_engine import run_ticker_analysis
        run_ticker_analysis(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 4 — 💼 PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════════════════
with tab_port:
    st.markdown('<div class="section-header">Real-Time Portfolio Tracker</div>', unsafe_allow_html=True)
    _tab_card("💼", "Your holdings, live prices, technicals, and news signals",
              "Live P&L, allocation pie, 1-year vs SPY. Per-position: RSI, 50/200MA, beta, volatility, "
              "Golden/Death Cross, and news signals from Live Feed matching your tickers.",
              "Run Correlations with your tickers to unlock correlation risk flags for your specific holdings.")
    from modules.portfolio_tracker import run_portfolio_tracker
    run_portfolio_tracker()

# ═══════════════════════════════════════════════════════════════════════════
# 5 — 🔬 BACKTESTER  (before Live Signals intentionally)
# ═══════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown('<div class="section-header">Strategy Backtester</div>', unsafe_allow_html=True)
    _tab_card("🔬", "Test strategies on history before using them for real",
              "4 strategies: MA Crossover, RSI Mean Reversion, Momentum, ML Weekly RF. "
              "Walk-forward OOS only. Transaction costs, bootstrap Sharpe CI, buy-and-hold benchmark.",
              "Find the best parameters here, then take those exact parameters to Live Signals for today's trade setup.")
    from modules.backtest_engine import run_backtest
    run_backtest()

# ═══════════════════════════════════════════════════════════════════════════
# 6 — 🔮 LIVE TRADING SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
with tab_strat:
    st.markdown('<div class="section-header">Live Trading Signals</div>', unsafe_allow_html=True)
    _tab_card("🔮", "Apply backtested strategy to current market data",
              "Same logic as Backtester, applied to live prices. Entry, stop loss, take profit, "
              "risk/reward ratio, exact indicator values, and a 1-year chart with overlays.",
              "Use the parameters you validated in the Backtester tab — don't guess.")
    from modules.strategy_signals import run_strategy_signals
    run_strategy_signals()

# ═══════════════════════════════════════════════════════════════════════════
# 7 — 🧪 STRATEGY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown('<div class="section-header">Strategy Historical Simulator</div>', unsafe_allow_html=True)
    _tab_card("🧪", "Walk-forward simulation across multiple tickers, strategies, and horizons",
              "Tests all 4 strategies simultaneously. Shows accuracy %, p-values, equity curves, "
              "and a ranked table of statistically significant combinations.",
              "p < 0.05 means statistically significant. Focus Live Signals on those combinations only.")
    from modules.auto_evaluator import run_auto_evaluator
    run_auto_evaluator()

# ═══════════════════════════════════════════════════════════════════════════
# 8 — 📰 NEWS SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════
with tab_sent:
    st.markdown('<div class="section-header">News Article Sentiment Scorer</div>', unsafe_allow_html=True)
    _tab_card("📰", "Paste any article URL → AI scores 5 dimensions",
              "Political bias, truthfulness, propaganda, hype, and panic (0–1). "
              "Results feed into Portfolio Risk and AI Advisor.",
              "Test it with a known satire article — it should score high hype and low truthfulness.")
    urls_input = st.text_area("Article URLs — one per line", height=120,
        placeholder="https://www.reuters.com/...\nhttps://www.cnbc.com/...")
    if st.button("Score Articles →", key="run_news"):
        if not GROQ_API_KEY: _no_key()
        elif not urls_input.strip(): st.error("Paste at least one URL.")
        else:
            urls = [u.strip() for u in urls_input.strip().splitlines() if u.strip()]
            from modules.sentiment_engine import run_sentiment_analysis
            run_sentiment_analysis(urls=urls, groq_api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 9 — 🔍 NEWS ACCURACY
# ═══════════════════════════════════════════════════════════════════════════
with tab_nse:
    st.markdown('<div class="section-header">News Signal Accuracy Evaluator</div>', unsafe_allow_html=True)
    _tab_card("🔍", "Did the Live Feed signals actually predict market direction?",
              "Reads every BUY/SHORT from Live Feed, Ticker Signals, and Briefing history. "
              "Fetches real prices at signal time and at horizon. Accuracy by confidence level with p-values.",
              "You need 30+ evaluated signals for reliable conclusions. Run Live Feed daily for 2–3 weeks first.")
    from modules.news_signal_evaluator import run_news_signal_evaluator
    run_news_signal_evaluator()

# ═══════════════════════════════════════════════════════════════════════════
# 10 — 📈 SIGNAL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<div class="section-header">Signal Performance Track Record</div>', unsafe_allow_html=True)
    _tab_card("📈", "Complete signal accuracy — all sources, horizon-matched, shock-filtered",
              "Horizon-matched (trading days not calendar days). Reversal-aware (opposing signal closes trade early). "
              "Black swan filter (3× normal move excluded). Your full track record.",
              "This is your proof of edge. Share this tab's numbers when pitching the product.")
    from modules.signal_performance import run_signal_performance
    run_signal_performance()

# ═══════════════════════════════════════════════════════════════════════════
# 11 — 🧠 PROFITABILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════
with tab_spe:
    st.markdown('<div class="section-header">Signal Profitability Engine</div>', unsafe_allow_html=True)
    _tab_card("🧠", "Self-improving intelligence — which signals actually make money?",
              "Analyses every signal by type, market regime (bull/bear/sideways), confidence level, and ticker. "
              "Volatility-adjusted returns. Auto-ranks: ✅ Keep / ⚠️ Watch / ❌ Kill.",
              "The more signal history you build, the more accurate and actionable this becomes.")
    from modules.signal_profitability import run_signal_profitability
    run_signal_profitability()

# ═══════════════════════════════════════════════════════════════════════════
# 12 — 📊 CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab_corr:
    st.markdown('<div class="section-header">Stock Correlation Analytics Engine</div>', unsafe_allow_html=True)
    _tab_card("📊", "How do your assets move relative to each other?",
              "Pairwise correlation table, heatmap, rolling chart, and RF predictor. "
              "Results feed into Portfolio Risk, AI Advisor, and Portfolio Tracker risk flags.",
              "Enter your actual portfolio tickers to get correlation risk flags specific to your holdings.")
    col_in, col_tags = st.columns([3, 2])
    with col_in:
        tickers_input = st.text_input("Tickers", placeholder="TSLA, NVDA, AMD, BTC-USD, ^GSPC, SPY")
    with col_tags:
        st.markdown('<div style="margin-top:1.5rem;"><span class="tag">STOCKS</span><span class="tag">ETFs</span><span class="tag">CRYPTO</span><span class="tag">INDICES</span></div>', unsafe_allow_html=True)
    if st.button("Run Correlation Analysis →", key="run_corr"):
        if not tickers_input.strip(): st.error("Enter at least 2 tickers.")
        else:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            if len(tickers) < 2: st.error("Need at least 2 tickers.")
            else:
                from modules.correlation_engine import run_correlation_analysis
                run_correlation_analysis(tickers=tickers, start_date=str(start_date),
                    rolling_window=rolling_window, n_estimators=n_estimators,
                    train_split=train_split/100)

# ═══════════════════════════════════════════════════════════════════════════
# 13 — ⚠ RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.markdown('<div class="section-header">Portfolio Risk Analysis</div>', unsafe_allow_html=True)
    _tab_card("⚠", "Correlation + sentiment → composite risk score",
              "Flags high correlation clusters, natural hedges, high hype + low truthfulness, panic signals. "
              "Composite 0–1 score: LOW / MODERATE / HIGH.",
              "Run Correlations and News Sentiment first to get the full analysis.")
    from modules.portfolio_risk import run_portfolio_risk
    run_portfolio_risk()

# ═══════════════════════════════════════════════════════════════════════════
# 14 — 🤖 AI ADVISOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_adv:
    st.markdown('<div class="section-header">AI Portfolio Advisor</div>', unsafe_allow_html=True)
    _tab_card("🤖", "Conversational AI grounded in your actual computed data",
              "Reads your real correlations, sentiment scores, and rolling trends as context — not generic advice. "
              "Structured sections: Insights, Risk, Recommendations, Outlook.",
              "Run Correlations and News Sentiment first — more data = richer advice.")
    if not GROQ_API_KEY: _no_key()
    else:
        from modules.advisor_engine import run_advisor
        run_advisor(groq_api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════════════
# 15 — 📜 SIGNAL LOG
# ═══════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown('<div class="section-header">Signal History Log</div>', unsafe_allow_html=True)
    _tab_card("📜", "Every signal from every tab, saved automatically",
              "All BUY/SHORT/HOLD/WATCH signals auto-logged to SQLite. Filterable by time, source, action, ticker. "
              "Charts, full table, CSV download.")
    from modules.signal_history import run_signal_history
    run_signal_history()

# ═══════════════════════════════════════════════════════════════════════════
# 16 — 📧 EMAIL ALERTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_email:
    st.markdown('<div class="section-header">Email Alert Configuration</div>', unsafe_allow_html=True)
    _tab_card("📧", "Automatic email alerts on HIGH urgency signals",
              "Configure Gmail (App Password) or SendGrid. HIGH urgency Live Feed signals "
              "send a formatted email with article, market impact, and all recommendations automatically.")
    from modules.email_alerts import render_email_setup
    render_email_setup()

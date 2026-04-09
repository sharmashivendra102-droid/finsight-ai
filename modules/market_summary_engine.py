"""
Market Open Summary Engine
===========================
Generates a structured pre-market briefing:
  - Overnight news from RSS feeds
  - Pre-market price moves via yfinance
  - Key economic events (static calendar awareness)
  - AI-generated sector-by-sector outlook
  - Top 3 tickers to watch with reasoning
  - Overall market bias for the day
"""

import streamlit as st
import feedparser
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

RSS_FEEDS = {
    "Reuters Business":  "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Markets":   "https://feeds.reuters.com/reuters/UKmarkets",
    "CNBC Top News":     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Finance":      "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "Yahoo Finance":     "https://finance.yahoo.com/news/rssindex",
    "MarketWatch":       "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "Investing.com":     "https://www.investing.com/rss/news.rss",
    "WSJ Markets":       "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
}

# Key market indicators to pull pre-market data for
MARKET_INDICATORS = {
    "S&P 500":   "^GSPC",
    "Nasdaq":    "^IXIC",
    "Dow Jones": "^DJI",
    "VIX":       "^VIX",
    "10Y Yield": "^TNX",
    "Gold":      "GLD",
    "Oil (WTI)": "CL=F",
    "BTC":       "BTC-USD",
}

SECTOR_ETFS = {
    "Technology":          "XLK",
    "Energy":              "XLE",
    "Financials":          "XLF",
    "Healthcare":          "XLV",
    "Industrials":         "XLI",
    "Consumer Disc.":      "XLY",
    "Consumer Staples":    "XLP",
    "Materials":           "XLB",
    "Real Estate":         "XLRE",
    "Utilities":           "XLU",
    "Communication Svcs":  "XLC",
}


@st.cache_data(ttl=300, show_spinner=False)   # cache 5 min — pre-market data doesn't change fast
def _fetch_market_data() -> dict:
    """Fetch latest price data for key indicators."""
    import yfinance as yf
    results = {}

    all_symbols = list(MARKET_INDICATORS.values()) + list(SECTOR_ETFS.values())

    for symbol in all_symbols:
        try:
            t   = yf.Ticker(symbol)
            hist = t.history(period="5d", interval="1d")
            if hist.empty or len(hist) < 2:
                continue
            close_today = float(hist["Close"].iloc[-1])
            close_prev  = float(hist["Close"].iloc[-2])
            pct_chg     = (close_today - close_prev) / close_prev * 100
            results[symbol] = {
                "price":   close_today,
                "prev":    close_prev,
                "pct_chg": pct_chg,
            }
        except Exception:
            continue

    return results


@st.cache_data(ttl=180, show_spinner=False)   # cache 3 min
def _fetch_overnight_news() -> list:
    """Pull latest articles from RSS feeds."""
    articles = []
    seen = set()
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                link    = entry.get("link", "")
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                summary = re.sub(r"<[^>]+>", " ", summary)
                summary = re.sub(r"\s+", " ", summary).strip()
                if not link or not title or link in seen:
                    continue
                seen.add(link)
                articles.append({
                    "source":  source,
                    "title":   title,
                    "summary": summary[:500],
                    "url":     link,
                })
        except Exception:
            continue
    return articles[:25]


def _generate_summary_with_groq(
    articles: list,
    market_data: dict,
    api_key: str,
    watchlist: list,
) -> dict:
    from groq import Groq

    # Build market data context
    market_lines = []
    for name, symbol in MARKET_INDICATORS.items():
        if symbol in market_data:
            d   = market_data[symbol]
            chg = d["pct_chg"]
            arrow = "▲" if chg >= 0 else "▼"
            market_lines.append(f"  {name} ({symbol}): {arrow} {abs(chg):.2f}%  |  Last: {d['price']:.2f}")

    sector_lines = []
    for name, symbol in SECTOR_ETFS.items():
        if symbol in market_data:
            d   = market_data[symbol]
            chg = d["pct_chg"]
            arrow = "▲" if chg >= 0 else "▼"
            sector_lines.append(f"  {name} ({symbol}): {arrow} {abs(chg):.2f}%")

    article_text = ""
    for i, art in enumerate(articles[:18], 1):
        article_text += f"[{i}] {art['source']}: {art['title']}\n{art['summary'][:300]}\n\n"

    watchlist_str = ", ".join(watchlist) if watchlist else "none specified"
    today_str     = datetime.now().strftime("%A, %B %d, %Y")
    time_str      = datetime.now().strftime("%H:%M ET")

    prompt = f"""You are the head of market research at a major investment bank.
Today is {today_str}. Time: {time_str}.
Generate a structured pre-market briefing for traders and investors.

MARKET DATA (latest):
{chr(10).join(market_lines) if market_lines else "Data unavailable"}

SECTOR PERFORMANCE:
{chr(10).join(sector_lines) if sector_lines else "Data unavailable"}

OVERNIGHT NEWS:
{article_text}

USER WATCHLIST: {watchlist_str}

Return ONLY a valid JSON object with this exact structure:
{{
  "market_bias": "<BULLISH|BEARISH|NEUTRAL|MIXED>",
  "bias_confidence": "<HIGH|MEDIUM|LOW>",
  "one_line_summary": "<single punchy sentence capturing today's market mood>",
  "key_themes": [
    {{
      "theme": "<theme name e.g. Geopolitical Risk, AI Earnings, Rate Expectations>",
      "description": "<2 sentences explaining this theme and its market impact>",
      "impact": "<POSITIVE|NEGATIVE|NEUTRAL>",
      "affected_sectors": ["<sector>"]
    }}
  ],
  "sector_outlook": [
    {{
      "sector": "<sector name>",
      "bias": "<BULLISH|BEARISH|NEUTRAL>",
      "reasoning": "<one sentence>",
      "etf": "<ETF symbol>"
    }}
  ],
  "tickers_to_watch": [
    {{
      "ticker": "<symbol>",
      "company": "<name>",
      "action": "<BUY|SHORT|HOLD|WATCH>",
      "confidence": "<HIGH|MEDIUM|LOW>",
      "reasoning": "<specific reason citing news or data>",
      "catalyst": "<what specific event or news is the catalyst>",
      "risk": "<key risk to this thesis>"
    }}
  ],
  "watchlist_signals": [
    {{
      "ticker": "<symbol from user watchlist>",
      "signal": "<BUY|SHORT|HOLD|WATCH>",
      "confidence": "<HIGH|MEDIUM|LOW>",
      "notes": "<brief note — can be LOW confidence if no direct news>"
    }}
  ],
  "macro_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "levels_to_watch": [
    {{
      "instrument": "<e.g. S&P 500, 10Y Yield, Oil>",
      "level": "<price or yield level>",
      "significance": "<why this level matters>"
    }}
  ],
  "morning_playbook": "<3-4 sentence actionable summary of what a trader should be thinking about today>"
}}

Rules:
- tickers_to_watch: exactly 3-5 tickers, only where you have HIGH or MEDIUM confidence
- sector_outlook: cover 4-6 sectors most affected by today's news
- key_themes: 2-4 themes maximum
- watchlist_signals: include ALL tickers from user watchlist, use LOW confidence if no direct news
- Be specific — cite actual news headlines, actual price levels
- Return ONLY valid JSON, no markdown, no backticks"""

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)
    except json.JSONDecodeError:
        st.error("❌ AI returned malformed response. Try regenerating.")
        return {}
    except Exception as e:
        st.error(f"❌ Groq error: {str(e)[:150]}")
        return {}


def _render_market_indicator(name, symbol, data):
    if symbol not in data:
        return
    d     = data[symbol]
    chg   = d["pct_chg"]
    color = "#4ade80" if chg >= 0 else "#f87171"
    arrow = "▲" if chg >= 0 else "▼"
    st.markdown(f"""
    <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;
                padding:0.6rem 0.8rem;text-align:center;">
        <div style="color:#6b8fad;font-size:0.7rem;font-family:'Space Mono',monospace;">{name}</div>
        <div style="color:#c9d8e8;font-size:0.85rem;font-weight:600;">{d['price']:.2f}</div>
        <div style="color:{color};font-size:0.8rem;font-weight:700;">{arrow} {abs(chg):.2f}%</div>
    </div>
    """, unsafe_allow_html=True)


def run_market_summary(api_key: str):

    today_str = datetime.now().strftime("%A, %B %d, %Y")
    time_str  = datetime.now().strftime("%H:%M")

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
        <div>
            <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#38bdf8;font-weight:700;">
                📋 Market Briefing
            </div>
            <div style="color:#6b8fad;font-size:0.85rem;">{today_str} · Generated {time_str} ET</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Watchlist input ────────────────────────────────────────────────────
    with st.expander("⚙️ Configure your watchlist (optional)", expanded=False):
        watchlist_input = st.text_input(
            "Your tickers (comma-separated) — these get specific signals in the briefing",
            placeholder="e.g. TSLA, NVDA, AAPL",
            key="summary_watchlist"
        )
        watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()] if watchlist_input else []

    col_gen, col_cache = st.columns([2, 3])
    with col_gen:
        generate_btn = st.button("🔄 Generate / Refresh Briefing", key="gen_summary",
                                 use_container_width=True)
    with col_cache:
        st.caption("Briefing is cached for 5 minutes. Click refresh to pull the latest news.")

    # Auto-show last briefing
    if not generate_btn and st.session_state.get("market_summary_data"):
        _display_summary(
            st.session_state["market_summary_data"],
            st.session_state.get("market_summary_market_data", {}),
            st.session_state.get("market_summary_articles", []),
        )
        return

    if not generate_btn:
        st.info("👆 Click **Generate Briefing** to get your pre-market summary.")
        return

    # ── Fetch data ─────────────────────────────────────────────────────────
    with st.spinner("📡 Fetching market data…"):
        market_data = _fetch_market_data()

    with st.spinner("📰 Pulling overnight news…"):
        articles = _fetch_overnight_news()

    st.caption(f"Fetched {len(articles)} articles from {len(RSS_FEEDS)} sources")

    with st.spinner("🧠 AI generating briefing…"):
        summary = _generate_summary_with_groq(articles, market_data, api_key, watchlist)

    if not summary:
        return

    st.session_state["market_summary_data"]        = summary
    st.session_state["market_summary_market_data"] = market_data
    st.session_state["market_summary_articles"]    = articles
    st.session_state["market_summary_time"]        = datetime.now().strftime("%H:%M")

    _display_summary(summary, market_data, articles)


def _display_summary(summary: dict, market_data: dict, articles: list):

    bias      = summary.get("market_bias", "NEUTRAL")
    bias_conf = summary.get("bias_confidence", "LOW")
    one_liner = summary.get("one_line_summary", "")

    bias_color = {"BULLISH": "#4ade80", "BEARISH": "#f87171",
                  "NEUTRAL": "#94a3b8", "MIXED": "#fbbf24"}.get(bias, "#94a3b8")
    bias_icon  = {"BULLISH": "🟢", "BEARISH": "🔴",
                  "NEUTRAL": "⚪", "MIXED": "🟡"}.get(bias, "⚪")
    conf_icon  = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔘"}.get(bias_conf, "🔘")

    # ── Market bias banner ─────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);
                border:2px solid {bias_color}33;border-radius:14px;
                padding:1.2rem 1.6rem;margin-bottom:1rem;">
        <div style="display:flex;align-items:center;gap:1rem;">
            <div style="font-family:'Space Mono',monospace;font-size:2rem;
                        font-weight:700;color:{bias_color};">
                {bias_icon} {bias}
            </div>
            <div>
                <div style="color:#c9d8e8;font-size:1rem;font-weight:500;">{one_liner}</div>
                <div style="color:#6b8fad;font-size:0.78rem;margin-top:0.2rem;">
                    {conf_icon} {bias_conf} confidence
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Market indicators strip ────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Market Snapshot</div>', unsafe_allow_html=True)
    ind_cols = st.columns(len(MARKET_INDICATORS))
    for i, (name, symbol) in enumerate(MARKET_INDICATORS.items()):
        with ind_cols[i]:
            _render_market_indicator(name, symbol, market_data)

    # ── Sector heatmap ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗂️ Sector Performance</div>', unsafe_allow_html=True)
    sec_cols = st.columns(6)
    for i, (name, symbol) in enumerate(SECTOR_ETFS.items()):
        with sec_cols[i % 6]:
            _render_market_indicator(name, symbol, market_data)

    # ── Key themes ─────────────────────────────────────────────────────────
    themes = summary.get("key_themes", [])
    if themes:
        st.markdown('<div class="section-header">🎯 Key Themes Today</div>', unsafe_allow_html=True)
        t_cols = st.columns(min(len(themes), 3))
        impact_color = {"POSITIVE": "#4ade80", "NEGATIVE": "#f87171", "NEUTRAL": "#94a3b8"}
        for i, theme in enumerate(themes):
            with t_cols[i % 3]:
                ic = impact_color.get(theme.get("impact", "NEUTRAL"), "#94a3b8")
                sectors_str = ", ".join(theme.get("affected_sectors", []))
                st.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid {ic}55;border-radius:10px;
                            padding:0.9rem 1rem;height:100%;">
                    <div style="font-weight:700;color:{ic};font-size:0.9rem;margin-bottom:0.4rem;">
                        {theme.get('theme','')}
                    </div>
                    <div style="color:#c9d8e8;font-size:0.82rem;line-height:1.5;">
                        {theme.get('description','')}
                    </div>
                    {"<div style='color:#6b8fad;font-size:0.75rem;margin-top:0.4rem;'>📂 " + sectors_str + "</div>" if sectors_str else ""}
                </div>
                """, unsafe_allow_html=True)

    # ── Tickers to watch ───────────────────────────────────────────────────
    tickers_to_watch = summary.get("tickers_to_watch", [])
    if tickers_to_watch:
        st.markdown('<div class="section-header">🎯 Top Tickers to Watch Today</div>', unsafe_allow_html=True)
        for tw in tickers_to_watch:
            action    = tw.get("action", "WATCH")
            conf      = tw.get("confidence", "LOW")
            action_colors = {"BUY": "🟢", "SHORT": "🔴", "HOLD": "🟡", "WATCH": "🔵"}
            conf_icon = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔘"}.get(conf, "🔘")
            dot       = action_colors.get(action, "⚪")
            emoji     = {"BUY": "📈", "SHORT": "📉", "HOLD": "➡️", "WATCH": "👁️"}.get(action, "👁️")

            tc1, tc2 = st.columns([1, 5])
            with tc1:
                st.markdown(f"**`{tw.get('ticker','?')}`**")
                st.caption(tw.get("company", ""))
                st.markdown(f"{dot} **{action}** {emoji}")
                st.caption(f"{conf_icon} {conf}")
            with tc2:
                st.markdown(f"**{tw.get('reasoning','')}**")
                st.caption(f"⚡ Catalyst: {tw.get('catalyst','')}")
                st.caption(f"⚠️ Risk: {tw.get('risk','')}")
            st.markdown("---")

    # ── Sector outlook ─────────────────────────────────────────────────────
    sector_outlook = summary.get("sector_outlook", [])
    if sector_outlook:
        st.markdown('<div class="section-header">🏭 Sector Outlook</div>', unsafe_allow_html=True)
        so_cols = st.columns(min(len(sector_outlook), 3))
        for i, so in enumerate(sector_outlook):
            sb = so.get("bias", "NEUTRAL")
            sc = {"BULLISH": "#4ade80", "BEARISH": "#f87171", "NEUTRAL": "#94a3b8"}.get(sb, "#94a3b8")
            si = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(sb, "⚪")
            with so_cols[i % 3]:
                st.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid {sc}44;border-radius:10px;
                            padding:0.8rem;margin-bottom:0.5rem;">
                    <div style="color:{sc};font-weight:700;font-size:0.88rem;">
                        {si} {so.get('sector','')}
                        <span style="color:#6b8fad;font-size:0.75rem;"> · {so.get('etf','')}</span>
                    </div>
                    <div style="color:#c9d8e8;font-size:0.8rem;margin-top:0.3rem;">
                        {so.get('reasoning','')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Watchlist signals ──────────────────────────────────────────────────
    watchlist_signals = summary.get("watchlist_signals", [])
    if watchlist_signals:
        st.markdown('<div class="section-header">📋 Your Watchlist Signals</div>', unsafe_allow_html=True)
        ws_cols = st.columns(min(len(watchlist_signals), 4))
        for i, ws in enumerate(watchlist_signals):
            sig    = ws.get("signal", "WATCH")
            conf   = ws.get("confidence", "LOW")
            dot    = {"BUY": "🟢", "SHORT": "🔴", "HOLD": "🟡", "WATCH": "🔵"}.get(sig, "⚪")
            ci     = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔘"}.get(conf, "🔘")
            with ws_cols[i % 4]:
                st.markdown(f"""
                <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;
                            padding:0.8rem;text-align:center;margin-bottom:0.5rem;">
                    <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
                                color:#38bdf8;font-weight:700;">{ws.get('ticker','?')}</div>
                    <div style="font-size:0.9rem;font-weight:700;">{dot} {sig}</div>
                    <div style="font-size:0.75rem;color:#6b8fad;">{ci} {conf}</div>
                    <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;">
                        {ws.get('notes','')[:80]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Levels to watch ────────────────────────────────────────────────────
    levels = summary.get("levels_to_watch", [])
    if levels:
        st.markdown('<div class="section-header">📏 Key Levels to Watch</div>', unsafe_allow_html=True)
        for lv in levels:
            st.markdown(f"**{lv.get('instrument','')} @ {lv.get('level','')}** — {lv.get('significance','')}")

    # ── Macro risks ────────────────────────────────────────────────────────
    risks = summary.get("macro_risks", [])
    if risks:
        st.markdown('<div class="section-header">⚠️ Macro Risks Today</div>', unsafe_allow_html=True)
        for r in risks:
            st.markdown(f"- {r}")

    # ── Morning playbook ───────────────────────────────────────────────────
    playbook = summary.get("morning_playbook", "")
    if playbook:
        st.markdown('<div class="section-header">☕ Morning Playbook</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#122338);
                    border-left:3px solid #38bdf8;border-radius:0 10px 10px 0;
                    padding:1.2rem 1.4rem;font-size:0.95rem;color:#c9d8e8;line-height:1.7;">
            {playbook}
        </div>
        """, unsafe_allow_html=True)

    # ── News feed used ─────────────────────────────────────────────────────
    with st.expander(f"📰 {len(articles)} articles used in this briefing"):
        for art in articles:
            st.markdown(f"- [{art['title']}]({art['url']}) — *{art['source']}*")

    gen_time = st.session_state.get("market_summary_time", "")
    st.caption(f"⏱ Briefing generated at {gen_time}  ·  ⚠️ Not financial advice")

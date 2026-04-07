"""
Live News Intelligence Engine
- Polls RSS feeds from major financial news sources
- Sends new articles to Groq for market interpretation
- Returns structured stock recommendations: BUY / SHORT / HOLD
- Tracks seen articles to avoid reprocessing
"""

import streamlit as st
import feedparser
import json
import re
import time
import hashlib
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ── RSS Feed sources — all free, no API key needed ────────────────────────────
RSS_FEEDS = {
    "Reuters Business":     "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Markets":      "https://feeds.reuters.com/reuters/UKmarkets",
    "CNBC Top News":        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Finance":         "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "Yahoo Finance":        "https://finance.yahoo.com/news/rssindex",
    "MarketWatch":          "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "Seeking Alpha":        "https://seekingalpha.com/feed.xml",
    "Investing.com":        "https://www.investing.com/rss/news.rss",
    "FT Markets":           "https://www.ft.com/rss/home/uk",
    "WSJ Markets":          "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
}

MAX_ARTICLES_PER_REFRESH = 5   # max new articles to process per cycle (Groq rate limit)
MAX_FEED_ARTICLES        = 3   # articles to pull from each feed


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _fetch_latest_articles() -> list:
    """Pull fresh articles from all RSS feeds. Returns list of dicts."""
    articles = []
    for source, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:MAX_FEED_ARTICLES]:
                url     = entry.get("link", "")
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                # Clean HTML from summary
                summary = re.sub(r"<[^>]+>", " ", summary)
                summary = re.sub(r"\s+", " ", summary).strip()

                published = entry.get("published", "")
                if not url or not title:
                    continue

                articles.append({
                    "id":        _article_id(url),
                    "source":    source,
                    "title":     title,
                    "summary":   summary[:800],
                    "url":       url,
                    "published": published,
                })
        except Exception:
            continue
    return articles


def _analyze_with_groq(articles: list, api_key: str) -> list:
    """
    Send batch of articles to Groq.
    Returns list of analysis dicts with stock recommendations.
    """
    from groq import Groq

    if not articles:
        return []

    # Build article list for prompt
    article_text = ""
    for i, art in enumerate(articles, 1):
        article_text += f"\n[{i}] SOURCE: {art['source']}\nTITLE: {art['title']}\nSUMMARY: {art['summary']}\nURL: {art['url']}\n"

    prompt = f"""You are a professional Wall Street trader and market analyst with expertise in interpreting breaking news and its market impact.

Analyze the following {len(articles)} news article(s) and for EACH one return a JSON object in a JSON array.

For each article return:
{{
  "article_index": <1-based index>,
  "headline_summary": "<1 sentence summary of what happened>",
  "market_impact": "<BULLISH|BEARISH|NEUTRAL>",
  "impact_reasoning": "<2-3 sentences explaining WHY this affects markets>",
  "urgency": "<HIGH|MEDIUM|LOW>",
  "recommendations": [
    {{
      "ticker": "<stock ticker or sector ETF e.g. NVDA, XLF, QQQ>",
      "action": "<BUY|SHORT|HOLD|WATCH>",
      "reasoning": "<specific reason for this ticker>",
      "confidence": "<HIGH|MEDIUM|LOW>",
      "time_horizon": "<INTRADAY|SWING (1-5 days)|MEDIUM (weeks)|LONG (months)>"
    }}
  ],
  "affected_sectors": ["<sector1>", "<sector2>"],
  "key_risk": "<main risk to this thesis>"
}}

Rules:
- Only recommend specific tickers or ETFs when you have high confidence the news directly impacts them
- If news is too vague or general to recommend specific stocks, say WATCH on broad ETFs like SPY or QQQ
- Be specific and cite the news in your reasoning
- If news is not market-relevant, set market_impact to NEUTRAL and recommendations to []
- Return ONLY a valid JSON array, no markdown, no explanation

Articles to analyze:
{article_text}"""

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Extract JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        results = json.loads(raw)
        if not isinstance(results, list):
            results = [results]
        return results

    except json.JSONDecodeError:
        return []
    except Exception as e:
        st.warning(f"⚠️ Groq error during live analysis: {str(e)[:100]}")
        return []


def _render_signal_card(article: dict, analysis: dict):
    """Render a single news signal card."""
    impact = analysis.get("market_impact", "NEUTRAL")
    urgency = analysis.get("urgency", "LOW")
    recs = analysis.get("recommendations", [])

    # Colour by impact
    if impact == "BULLISH":
        border_color = "#166534"
        impact_color = "#4ade80"
        impact_icon  = "🟢"
    elif impact == "BEARISH":
        border_color = "#7f1d1d"
        impact_color = "#f87171"
        impact_icon  = "🔴"
    else:
        border_color = "#1e3a5f"
        impact_color = "#94a3b8"
        impact_icon  = "⚪"

    urgency_badge = {"HIGH": "🔥 HIGH", "MEDIUM": "⚡ MEDIUM", "LOW": "💤 LOW"}.get(urgency, urgency)
    urgency_color = {"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#6b8fad"}.get(urgency, "#6b8fad")

    # Build recommendations HTML
    recs_html = ""
    for rec in recs:
        action = rec.get("action", "WATCH")
        action_colors = {
            "BUY":   ("#4ade80", "#052e16"),
            "SHORT": ("#f87171", "#2d0a0a"),
            "HOLD":  ("#fbbf24", "#1c1200"),
            "WATCH": ("#7dd3fc", "#071827"),
        }
        ac, abg = action_colors.get(action, ("#94a3b8", "#0d1b2a"))
        confidence = rec.get("confidence", "")
        horizon = rec.get("time_horizon", "")
        ticker = rec.get("ticker", "")
        reasoning = rec.get("reasoning", "")

        recs_html += f"""
        <div style="background:{abg};border:1px solid {ac}33;border-radius:8px;padding:0.6rem 1rem;margin:0.4rem 0;display:flex;align-items:flex-start;gap:1rem;">
            <div style="min-width:80px;">
                <div style="font-family:'Space Mono',monospace;font-size:1rem;font-weight:700;color:{ac};">{ticker}</div>
                <div style="background:{ac};color:#000;border-radius:4px;padding:0.1rem 0.4rem;font-size:0.7rem;font-weight:700;font-family:'Space Mono',monospace;display:inline-block;margin-top:2px;">{action}</div>
            </div>
            <div style="flex:1;">
                <div style="color:#c9d8e8;font-size:0.83rem;">{reasoning}</div>
                <div style="color:#6b8fad;font-size:0.75rem;margin-top:0.3rem;">Confidence: {confidence} &nbsp;|&nbsp; Horizon: {horizon}</div>
            </div>
        </div>"""

    sectors = ", ".join(analysis.get("affected_sectors", []))
    key_risk = analysis.get("key_risk", "")
    published = article.get("published", "")
    now_str = datetime.now().strftime("%H:%M:%S")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#0f2035);border:1px solid {border_color};
                border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;">

        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.6rem;">
            <div style="flex:1;">
                <div style="color:#6b8fad;font-size:0.75rem;font-family:'Space Mono',monospace;">
                    {article['source']} &nbsp;·&nbsp; Detected {now_str}
                </div>
                <div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-top:0.3rem;line-height:1.4;">
                    {article['title']}
                </div>
            </div>
            <div style="text-align:right;margin-left:1rem;min-width:90px;">
                <div style="font-family:'Space Mono',monospace;font-size:0.85rem;font-weight:700;color:{impact_color};">
                    {impact_icon} {impact}
                </div>
                <div style="font-size:0.72rem;color:{urgency_color};margin-top:2px;">{urgency_badge}</div>
            </div>
        </div>

        <div style="color:#94a3b8;font-size:0.83rem;margin-bottom:0.8rem;line-height:1.5;">
            {analysis.get('impact_reasoning', '')}
        </div>

        {recs_html if recs_html else '<div style="color:#6b8fad;font-size:0.82rem;">No specific stock recommendations for this article.</div>'}

        <div style="margin-top:0.8rem;display:flex;gap:1rem;flex-wrap:wrap;">
            {"<div style='font-size:0.75rem;color:#6b8fad;'>📂 " + sectors + "</div>" if sectors else ""}
            {"<div style='font-size:0.75rem;color:#fbbf24;'>⚠️ Risk: " + key_risk + "</div>" if key_risk else ""}
        </div>

        <div style="margin-top:0.6rem;">
            <a href="{article['url']}" target="_blank"
               style="font-size:0.75rem;color:#38bdf8;text-decoration:none;">
               🔗 Read full article →
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def run_live_news(api_key: str, refresh_interval: int):
    """Main function called from the Live Intelligence tab."""

    # ── Init session state ─────────────────────────────────────────────────
    if "seen_article_ids"   not in st.session_state:
        st.session_state["seen_article_ids"]   = set()
    if "processed_signals"  not in st.session_state:
        st.session_state["processed_signals"]  = []   # list of (article, analysis)
    if "last_fetch_time"    not in st.session_state:
        st.session_state["last_fetch_time"]    = None
    if "live_running"       not in st.session_state:
        st.session_state["live_running"]       = False
    if "total_processed"    not in st.session_state:
        st.session_state["total_processed"]    = 0

    # ── Control bar ────────────────────────────────────────────────────────
    col_start, col_stop, col_clear, col_status = st.columns([1, 1, 1, 3])

    with col_start:
        if st.button("▶️ Start Feed", key="live_start", use_container_width=True):
            st.session_state["live_running"] = True

    with col_stop:
        if st.button("⏹️ Stop Feed", key="live_stop", use_container_width=True):
            st.session_state["live_running"] = False

    with col_clear:
        if st.button("🗑️ Clear", key="live_clear", use_container_width=True):
            st.session_state["processed_signals"] = []
            st.session_state["seen_article_ids"]  = set()
            st.session_state["total_processed"]   = 0
            st.rerun()

    with col_status:
        is_running = st.session_state["live_running"]
        status_color = "#4ade80" if is_running else "#6b8fad"
        status_text  = "● LIVE — scanning news sources" if is_running else "○ PAUSED"
        last_fetch   = st.session_state["last_fetch_time"]
        last_str     = f"Last scan: {last_fetch}" if last_fetch else "Not scanned yet"
        total        = st.session_state["total_processed"]
        st.markdown(f"""
        <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;
                    padding:0.6rem 1rem;display:flex;justify-content:space-between;align-items:center;">
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:{status_color};">{status_text}</div>
            <div style="font-size:0.75rem;color:#6b8fad;">{last_str} &nbsp;·&nbsp; {total} articles processed</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Filter controls ────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            filter_impact = st.multiselect(
                "Show impact types",
                ["BULLISH", "BEARISH", "NEUTRAL"],
                default=["BULLISH", "BEARISH"],
                key="filter_impact"
            )
        with fc2:
            filter_urgency = st.multiselect(
                "Show urgency levels",
                ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM"],
                key="filter_urgency"
            )

    st.markdown("---")

    # ── Fetch and process new articles ─────────────────────────────────────
    if st.session_state["live_running"]:
        with st.spinner("📡 Scanning news sources…"):
            all_articles = _fetch_latest_articles()

        # Filter to only unseen articles
        new_articles = [
            a for a in all_articles
            if a["id"] not in st.session_state["seen_article_ids"]
        ]

        if new_articles:
            # Take up to MAX_ARTICLES_PER_REFRESH
            batch = new_articles[:MAX_ARTICLES_PER_REFRESH]

            with st.spinner(f"🧠 AI analysing {len(batch)} new article(s)…"):
                analyses = _analyze_with_groq(batch, api_key)

            # Match analyses back to articles by index
            for analysis in analyses:
                idx = analysis.get("article_index", 1) - 1
                if 0 <= idx < len(batch):
                    article = batch[idx]
                    st.session_state["seen_article_ids"].add(article["id"])
                    st.session_state["processed_signals"].insert(0, (article, analysis))
                    st.session_state["total_processed"] += 1

            # Mark remaining unseen articles as seen without processing
            for a in new_articles[MAX_ARTICLES_PER_REFRESH:]:
                st.session_state["seen_article_ids"].add(a["id"])

        st.session_state["last_fetch_time"] = datetime.now().strftime("%H:%M:%S")

    # ── Render signals ─────────────────────────────────────────────────────
    signals = st.session_state["processed_signals"]

    if not signals:
        if st.session_state["live_running"]:
            st.info("⏳ Scanning… first results will appear shortly.")
        else:
            st.info("▶️ Press **Start Feed** to begin scanning live news sources.")
        return

    # Apply filters
    filtered = []
    for article, analysis in signals:
        impact  = analysis.get("market_impact", "NEUTRAL")
        urgency = analysis.get("urgency", "LOW")
        fi = st.session_state.get("filter_impact",  ["BULLISH", "BEARISH"])
        fu = st.session_state.get("filter_urgency", ["HIGH", "MEDIUM"])
        if impact in fi and urgency in fu:
            filtered.append((article, analysis))

    if not filtered:
        st.info("No signals match your current filters.")
        return

    st.markdown(f"**{len(filtered)} signal(s)** — newest first")
    st.markdown("")

    for article, analysis in filtered:
        _render_signal_card(article, analysis)

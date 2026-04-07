"""
Live News Intelligence Engine
- Polls RSS feeds from major financial news sources
- Sends new articles to Groq for market interpretation
- Returns structured BUY / SHORT / HOLD signals
"""

import streamlit as st
import feedparser
import json
import re
import hashlib
from datetime import datetime
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
    "FT":                "https://www.ft.com/rss/home/uk",
    "Seeking Alpha":     "https://seekingalpha.com/feed.xml",
}

MAX_ARTICLES_PER_REFRESH = 5
MAX_FEED_ARTICLES        = 3


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _fetch_latest_articles() -> list:
    articles = []
    for source, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:MAX_FEED_ARTICLES]:
                url     = entry.get("link", "")
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                summary = re.sub(r"<[^>]+>", " ", summary)
                summary = re.sub(r"\s+", " ", summary).strip()
                if not url or not title:
                    continue
                articles.append({
                    "id":      _article_id(url),
                    "source":  source,
                    "title":   title,
                    "summary": summary[:800],
                    "url":     url,
                })
        except Exception:
            continue
    return articles


def _analyze_with_groq(articles: list, api_key: str) -> list:
    from groq import Groq

    if not articles:
        return []

    article_text = ""
    for i, art in enumerate(articles, 1):
        article_text += f"\n[{i}] SOURCE: {art['source']}\nTITLE: {art['title']}\nSUMMARY: {art['summary']}\n"

    prompt = f"""You are a professional Wall Street trader. Analyze these {len(articles)} news articles and return a JSON array.

For each article return exactly this structure:
{{
  "article_index": <number>,
  "headline_summary": "<one sentence>",
  "market_impact": "<BULLISH|BEARISH|NEUTRAL>",
  "impact_reasoning": "<2-3 sentences>",
  "urgency": "<HIGH|MEDIUM|LOW>",
  "recommendations": [
    {{
      "ticker": "<e.g. NVDA, XLE, QQQ>",
      "action": "<BUY|SHORT|HOLD|WATCH>",
      "reasoning": "<why this ticker>",
      "confidence": "<HIGH|MEDIUM|LOW>",
      "time_horizon": "<INTRADAY|SWING (1-5 days)|MEDIUM (weeks)|LONG (months)>"
    }}
  ],
  "affected_sectors": ["<sector>"],
  "key_risk": "<main risk>"
}}

Return ONLY a valid JSON array. No markdown, no backticks, no explanation.

Articles:
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
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        results = json.loads(raw)
        if not isinstance(results, list):
            results = [results]
        return results
    except Exception as e:
        st.warning(f"⚠️ Groq error: {str(e)[:100]}")
        return []


def _render_signal_card(article: dict, analysis: dict):
    """Render a signal card using native Streamlit components — no raw HTML."""

    impact  = analysis.get("market_impact", "NEUTRAL")
    urgency = analysis.get("urgency", "LOW")
    recs    = analysis.get("recommendations", [])

    # ── Impact colour mapping ──────────────────────────────────────────────
    impact_icon  = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(impact, "⚪")
    urgency_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}.get(urgency, "")
    action_emoji = {"BUY": "📈", "SHORT": "📉", "HOLD": "➡️", "WATCH": "👁️"}

    # ── Card container ─────────────────────────────────────────────────────
    with st.container():
        st.markdown("---")

        # Header row
        h_col1, h_col2 = st.columns([5, 1])
        with h_col1:
            st.caption(f"**{article['source']}** · Detected {datetime.now().strftime('%H:%M:%S')}")
            st.markdown(f"**{article['title']}**")
        with h_col2:
            st.markdown(f"**{impact_icon} {impact}**")
            st.caption(f"{urgency_icon} {urgency}")

        # Reasoning
        st.markdown(f"> {analysis.get('impact_reasoning', '')}")

        # Recommendations
        if recs:
            rec_cols = st.columns(min(len(recs), 3))
            for i, rec in enumerate(recs):
                action     = rec.get("action", "WATCH")
                ticker     = rec.get("ticker", "—")
                reasoning  = rec.get("reasoning", "")
                confidence = rec.get("confidence", "")
                horizon    = rec.get("time_horizon", "")
                emoji      = action_emoji.get(action, "👁️")

                action_colors = {
                    "BUY":   "🟢",
                    "SHORT": "🔴",
                    "HOLD":  "🟡",
                    "WATCH": "🔵",
                }
                dot = action_colors.get(action, "⚪")

                with rec_cols[i % 3]:
                    st.markdown(f"""
**{dot} {ticker}** — `{action}` {emoji}

{reasoning}

*Confidence: {confidence} · {horizon}*
""")

        # Sectors + risk + link
        sectors  = ", ".join(analysis.get("affected_sectors", []))
        key_risk = analysis.get("key_risk", "")
        meta_parts = []
        if sectors:
            meta_parts.append(f"📂 {sectors}")
        if key_risk:
            meta_parts.append(f"⚠️ Risk: {key_risk}")
        if meta_parts:
            st.caption("  ·  ".join(meta_parts))

        st.markdown(f"[🔗 Read full article →]({article['url']})")


def run_live_news(api_key: str, refresh_interval: int):

    # ── Init session state ─────────────────────────────────────────────────
    if "seen_article_ids"  not in st.session_state:
        st.session_state["seen_article_ids"]  = set()
    if "processed_signals" not in st.session_state:
        st.session_state["processed_signals"] = []
    if "last_fetch_time"   not in st.session_state:
        st.session_state["last_fetch_time"]   = None
    if "live_running"      not in st.session_state:
        st.session_state["live_running"]      = False
    if "total_processed"   not in st.session_state:
        st.session_state["total_processed"]   = 0

    # ── Control bar ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    with c1:
        if st.button("▶️ Start", key="live_start", use_container_width=True):
            st.session_state["live_running"] = True
            st.rerun()
    with c2:
        if st.button("⏹️ Stop", key="live_stop", use_container_width=True):
            st.session_state["live_running"] = False
    with c3:
        if st.button("🗑️ Clear", key="live_clear", use_container_width=True):
            st.session_state["processed_signals"] = []
            st.session_state["seen_article_ids"]  = set()
            st.session_state["total_processed"]   = 0
            st.rerun()
    with c4:
        is_running = st.session_state["live_running"]
        last_fetch = st.session_state["last_fetch_time"]
        total      = st.session_state["total_processed"]
        status     = "🟢 LIVE — scanning news sources" if is_running else "⚪ PAUSED"
        last_str   = f"Last scan: {last_fetch}" if last_fetch else "Not scanned yet"
        st.info(f"{status}  ·  {last_str}  ·  {total} articles processed")

    # ── Filters ────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters"):
        f1, f2 = st.columns(2)
        with f1:
            filter_impact = st.multiselect("Impact", ["BULLISH", "BEARISH", "NEUTRAL"],
                                           default=["BULLISH", "BEARISH"], key="filter_impact")
        with f2:
            filter_urgency = st.multiselect("Urgency", ["HIGH", "MEDIUM", "LOW"],
                                            default=["HIGH", "MEDIUM"], key="filter_urgency")

    # ── Fetch & process ────────────────────────────────────────────────────
    if st.session_state["live_running"]:
        with st.spinner("📡 Scanning news sources…"):
            all_articles = _fetch_latest_articles()

        new_articles = [
            a for a in all_articles
            if a["id"] not in st.session_state["seen_article_ids"]
        ]

        if new_articles:
            batch = new_articles[:MAX_ARTICLES_PER_REFRESH]
            with st.spinner(f"🧠 AI analysing {len(batch)} new article(s)…"):
                analyses = _analyze_with_groq(batch, api_key)

            for analysis in analyses:
                idx = analysis.get("article_index", 1) - 1
                if 0 <= idx < len(batch):
                    article = batch[idx]
                    st.session_state["seen_article_ids"].add(article["id"])
                    st.session_state["processed_signals"].insert(0, (article, analysis))
                    st.session_state["total_processed"] += 1

            for a in new_articles[MAX_ARTICLES_PER_REFRESH:]:
                st.session_state["seen_article_ids"].add(a["id"])

        st.session_state["last_fetch_time"] = datetime.now().strftime("%H:%M:%S")

    # ── Render signals ─────────────────────────────────────────────────────
    signals = st.session_state["processed_signals"]

    if not signals:
        if st.session_state["live_running"]:
            st.info("⏳ Scanning… first results will appear shortly.")
        else:
            st.info("Press **▶️ Start** to begin scanning live news sources.")
        return

    filtered = [
        (art, ana) for art, ana in signals
        if ana.get("market_impact", "NEUTRAL") in filter_impact
        and ana.get("urgency", "LOW") in filter_urgency
    ]

    if not filtered:
        st.info("No signals match your current filters.")
        return

    st.caption(f"**{len(filtered)} signal(s)** — newest first")

    for article, analysis in filtered:
        _render_signal_card(article, analysis)

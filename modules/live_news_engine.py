"""
Live News Intelligence Engine
- Polls RSS feeds from major financial news sources
- User-controlled article age filter
- Sends new articles to Groq for market interpretation
- Returns structured BUY / SHORT / HOLD signals
"""

import streamlit as st
import feedparser
import re
import json
import hashlib
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
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
    "Seeking Alpha":     "https://seekingalpha.com/feed.xml",
    "FT":                "https://www.ft.com/rss/home/uk",
}

MAX_ARTICLES_PER_REFRESH = 5
MAX_FEED_ARTICLES        = 5


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _parse_pub_date(entry) -> datetime | None:
    """Try multiple date fields, return UTC-aware datetime or None."""
    for field in ("published", "updated", "created"):
        raw = entry.get(field, "")
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _fetch_latest_articles(max_age_hours: int) -> list:
    """Fetch articles from RSS feeds, filtering by age."""
    articles = []
    now      = datetime.now(timezone.utc)
    cutoff   = now - timedelta(hours=max_age_hours)

    for source, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            count = 0
            for entry in feed.entries:
                if count >= MAX_FEED_ARTICLES:
                    break

                url     = entry.get("link", "")
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                summary = re.sub(r"<[^>]+>", " ", summary)
                summary = re.sub(r"\s+", " ", summary).strip()
                if not url or not title:
                    continue

                # Age filter
                pub_dt = _parse_pub_date(entry)
                if pub_dt is not None:
                    if pub_dt < cutoff:
                        continue   # article is too old
                    age_str = _format_age(now, pub_dt)
                else:
                    age_str = "unknown age"

                articles.append({
                    "id":       _article_id(url),
                    "source":   source,
                    "title":    title,
                    "summary":  summary[:800],
                    "url":      url,
                    "pub_dt":   pub_dt,
                    "age_str":  age_str,
                })
                count += 1

        except Exception:
            continue

    # Sort newest first
    articles.sort(key=lambda x: x["pub_dt"] or datetime.min.replace(tzinfo=timezone.utc),
                  reverse=True)
    return articles


def _format_age(now: datetime, pub_dt: datetime) -> str:
    diff = now - pub_dt
    mins = int(diff.total_seconds() / 60)
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def _analyze_with_groq(articles: list, api_key: str) -> list:
    import time
    from groq import Groq, RateLimitError, APIStatusError, APIConnectionError

    if not articles:
        return []

    article_text = ""
    for i, art in enumerate(articles, 1):
        article_text += f"\n[{i}] SOURCE: {art['source']}\nTITLE: {art['title']}\nSUMMARY: {art['summary']}\nURL: {art['url']}\n"

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
    messages = [{"role": "user", "content": prompt}]
    wait = 8

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
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

        except RateLimitError:
            if attempt < 3:
                st.toast(f"⏳ Live Feed engine busy — retrying in {wait}s ({attempt}/3)", icon="⚡")
                time.sleep(wait)
                wait *= 2
            else:
                st.warning(
                    "⚡ **Rate limit reached.** The Live News Feed is processing too many requests. "
                    "Wait 60 seconds, or reduce the auto-refresh interval in Feed Settings."
                )
                return []

        except APIConnectionError:
            if attempt < 3:
                st.toast(f"📡 Connection issue — retrying ({attempt}/3)…")
                time.sleep(4)
            else:
                st.error("📡 **Could not connect to the analysis engine.** Check your internet connection.")
                return []

        except APIStatusError as e:
            code = getattr(e, "status_code", "?")
            if code == 503 and attempt < 3:
                st.toast(f"🔄 Engine temporarily unavailable — retrying ({attempt}/3)…")
                time.sleep(10)
            elif code == 401:
                st.error("🔴 **Invalid API key.** Check your `GROQ_API_KEY` in Streamlit Secrets.")
                return []
            else:
                if attempt < 3:
                    time.sleep(wait); wait *= 2
                else:
                    st.error(f"🔴 **Live Feed engine error (HTTP {code})** — {str(e)[:120]}")
                    return []

        except json.JSONDecodeError:
            st.warning("⚠️ Live Feed received a malformed response — skipping this batch.")
            return []

        except Exception as e:
            if attempt < 3:
                time.sleep(wait); wait *= 2
            else:
                st.warning(f"⚠️ Live Feed error: {str(e)[:120]}")
                return []

    return []


def _render_signal_card(article: dict, analysis: dict):
    impact  = analysis.get("market_impact", "NEUTRAL")
    urgency = analysis.get("urgency", "LOW")
    recs    = analysis.get("recommendations", [])

    impact_icon  = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(impact, "⚪")
    urgency_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}.get(urgency, "")
    action_emoji = {"BUY": "📈", "SHORT": "📉", "HOLD": "➡️", "WATCH": "👁️"}
    action_dot   = {"BUY": "🟢", "SHORT": "🔴", "HOLD": "🟡", "WATCH": "🔵"}

    with st.container():
        st.markdown("---")
        h1, h2 = st.columns([5, 1])
        with h1:
            age = article.get("age_str", "")
            st.caption(f"**{article['source']}** · {age} · Detected {datetime.now().strftime('%H:%M:%S')}")
            st.markdown(f"**{article['title']}**")
        with h2:
            st.markdown(f"**{impact_icon} {impact}**")
            st.caption(f"{urgency_icon} {urgency}")

        st.markdown(f"> {analysis.get('impact_reasoning', '')}")

        if recs:
            rec_cols = st.columns(min(len(recs), 3))
            for i, rec in enumerate(recs):
                action    = rec.get("action", "WATCH")
                ticker    = rec.get("ticker", "—")
                reasoning = rec.get("reasoning", "")
                conf      = rec.get("confidence", "")
                horizon   = rec.get("time_horizon", "")
                dot       = action_dot.get(action, "⚪")
                emoji     = action_emoji.get(action, "👁️")
                with rec_cols[i % 3]:
                    st.markdown(f"**{dot} {ticker}** — `{action}` {emoji}\n\n{reasoning}\n\n*{conf} · {horizon}*")

        sectors  = ", ".join(analysis.get("affected_sectors", []))
        key_risk = analysis.get("key_risk", "")
        parts = []
        if sectors:  parts.append(f"📂 {sectors}")
        if key_risk: parts.append(f"⚠️ {key_risk}")
        if parts:    st.caption("  ·  ".join(parts))
        st.markdown(f"[🔗 Read full article →]({article['url']})")


def run_live_news(api_key: str, refresh_interval: int, max_age_hours: int):

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

    # ── Controls ───────────────────────────────────────────────────────────
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
        status     = "🟢 LIVE" if is_running else "⚪ PAUSED"
        last_str   = f"Last scan: {last_fetch}" if last_fetch else "Not scanned yet"
        st.info(f"{status}  ·  {last_str}  ·  {total} processed  ·  Max article age: **{max_age_hours}h**")

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
            all_articles = _fetch_latest_articles(max_age_hours)

        if not all_articles:
            st.warning(f"⚠️ No articles found within the last {max_age_hours} hour(s). Try increasing the Max Article Age in the sidebar.")

        new_articles = [
            a for a in all_articles
            if a["id"] not in st.session_state["seen_article_ids"]
        ]

        if new_articles:
            batch = new_articles[:MAX_ARTICLES_PER_REFRESH]
            with st.spinner(f"🧠 Analysing {len(batch)} new article(s)…"):
                analyses = _analyze_with_groq(batch, api_key)

            for analysis in analyses:
                idx = analysis.get("article_index", 1) - 1
                if 0 <= idx < len(batch):
                    article = batch[idx]
                    st.session_state["seen_article_ids"].add(article["id"])
                    st.session_state["processed_signals"].insert(0, (article, analysis))
                    st.session_state["total_processed"] += 1

                    # Log to signal history
                    try:
                        from modules.signal_history import log_signals_from_live
                        log_signals_from_live(article, analysis)
                    except Exception:
                        pass

                    # Email alert for HIGH urgency
                    if analysis.get("urgency") == "HIGH":
                        try:
                            from modules.email_alerts import send_alert, is_configured
                            if is_configured():
                                sent = send_alert(article, analysis)
                                if sent:
                                    st.toast("📧 Alert emailed for HIGH urgency signal!")
                        except Exception:
                            pass

            for a in new_articles[MAX_ARTICLES_PER_REFRESH:]:
                st.session_state["seen_article_ids"].add(a["id"])

        st.session_state["last_fetch_time"] = datetime.now().strftime("%H:%M:%S")

    # ── Render ─────────────────────────────────────────────────────────────
    signals = st.session_state["processed_signals"]

    if not signals:
        if st.session_state["live_running"]:
            st.info("⏳ Scanning… results will appear shortly.")
        else:
            st.info("Press **▶️ Start** to begin scanning live news.")
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

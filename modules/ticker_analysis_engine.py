"""
Ticker Analysis Engine
- User enters specific ticker(s)
- System searches RSS feeds for relevant articles
- Groq analyses everything and returns per-ticker signals
- Same signal format as Live Intelligence but ticker-focused
"""

import streamlit as st
import feedparser
import re
import json
import hashlib
import requests
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
    "Seeking Alpha":     "https://seekingalpha.com/feed.xml",
    "FT":                "https://www.ft.com/rss/home/uk",
}

# Company name aliases so we can match articles that mention company
# names instead of tickers
TICKER_ALIASES = {
    "TSLA":  ["tesla", "elon musk", "tsla"],
    "NVDA":  ["nvidia", "nvda", "jensen huang"],
    "AAPL":  ["apple", "aapl", "tim cook", "iphone", "ipad", "macbook"],
    "MSFT":  ["microsoft", "msft", "satya nadella", "azure", "copilot"],
    "AMZN":  ["amazon", "amzn", "aws", "andy jassy"],
    "GOOGL": ["google", "googl", "alphabet", "sundar pichai", "gemini"],
    "META":  ["meta", "facebook", "instagram", "zuckerberg", "whatsapp"],
    "AMD":   ["amd", "advanced micro devices", "lisa su"],
    "INTC":  ["intel", "intc"],
    "NFLX":  ["netflix", "nflx"],
    "BABA":  ["alibaba", "baba"],
    "PDD":   ["pdd", "temu", "pinduoduo"],
    "BTC-USD": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "ETH-USD": ["ethereum", "eth", "crypto", "cryptocurrency"],
    "XLE":   ["energy", "oil", "crude", "petroleum", "exxon", "chevron"],
    "XLF":   ["banking", "finance", "jpmorgan", "goldman", "bank of america"],
    "QQQ":   ["nasdaq", "tech stocks", "technology"],
    "SPY":   ["s&p 500", "s&p500", "sp500", "market rally", "market crash"],
    "GLD":   ["gold", "precious metals"],
    "^GSPC": ["s&p 500", "s&p500", "market rally", "market crash"],
    "^DJI":  ["dow jones", "dow", "djia"],
    "^IXIC": ["nasdaq", "nasdaq composite"],
}


def _get_aliases(ticker: str) -> list:
    """Return search terms for a ticker — fallback to lowercase ticker itself."""
    ticker_upper = ticker.upper()
    base = TICKER_ALIASES.get(ticker_upper, [ticker.lower()])
    # Always include the raw ticker itself
    if ticker.lower() not in base:
        base = [ticker.lower()] + base
    return base


def _fetch_all_articles() -> list:
    """Fetch recent articles from all RSS feeds."""
    articles = []
    for source, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:
                url     = entry.get("link", "")
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                summary = re.sub(r"<[^>]+>", " ", summary)
                summary = re.sub(r"\s+", " ", summary).strip()
                if not url or not title:
                    continue
                articles.append({
                    "source":  source,
                    "title":   title,
                    "summary": summary[:600],
                    "url":     url,
                    "full_text": (title + " " + summary).lower(),
                })
        except Exception:
            continue
    return articles


def _filter_relevant(articles: list, tickers: list) -> list:
    """
    Return articles that mention any of the requested tickers or their aliases.
    Also always include top general market articles as macro context.
    """
    relevant = []
    seen_urls = set()

    all_terms = []
    for t in tickers:
        all_terms.extend(_get_aliases(t))

    # Ticker-specific articles first
    for art in articles:
        if art["url"] in seen_urls:
            continue
        text = art["full_text"]
        if any(term in text for term in all_terms):
            relevant.append({**art, "match_type": "direct"})
            seen_urls.add(art["url"])

    # Fill up with general macro context (up to 5 extra)
    macro_terms = ["s&p", "nasdaq", "market", "fed", "inflation",
                   "interest rate", "economy", "gdp", "recession",
                   "oil", "tariff", "trade", "earnings"]
    for art in articles:
        if art["url"] in seen_urls:
            continue
        if len(relevant) >= 15:
            break
        text = art["full_text"]
        if any(term in text for term in macro_terms):
            relevant.append({**art, "match_type": "macro"})
            seen_urls.add(art["url"])

    return relevant


def _analyze_tickers_with_groq(tickers: list, articles: list, api_key: str) -> list:
    from groq import Groq

    ticker_list = ", ".join(tickers)

    # Build article context
    article_text = ""
    for i, art in enumerate(articles[:12], 1):
        match_label = "DIRECT MATCH" if art.get("match_type") == "direct" else "MACRO CONTEXT"
        article_text += f"\n[{i}] [{match_label}] {art['source']}\nTITLE: {art['title']}\nSUMMARY: {art['summary']}\nURL: {art['url']}\n"

    prompt = f"""You are a professional Wall Street analyst and trader. The user wants a detailed analysis of these specific tickers: {ticker_list}

Below are recent news articles — some directly mention the ticker(s), others provide macro context.
Use ALL of them to form your analysis. Consider macro conditions even when articles don't directly mention the ticker.

IMPORTANT CONFIDENCE RULES:
- Use HIGH confidence ONLY when there is direct, clear news about the ticker
- Use MEDIUM confidence when inference is reasonable but indirect
- Use LOW confidence when you are extrapolating from macro context with little direct evidence
- Never inflate confidence — LOW confidence with a real signal is better than fake HIGH confidence
- If truly insufficient information exists for a ticker, still give your best LOW confidence signal based on macro context

For EACH ticker in [{ticker_list}], return a JSON object in an array with this exact structure:
{{
  "ticker": "<ticker symbol>",
  "company_name": "<full company or asset name>",
  "overall_signal": "<BUY|SHORT|HOLD|WATCH>",
  "overall_confidence": "<HIGH|MEDIUM|LOW>",
  "market_impact": "<BULLISH|BEARISH|NEUTRAL>",
  "urgency": "<HIGH|MEDIUM|LOW>",
  "summary": "<2-3 sentence overall assessment of this ticker right now>",
  "signals": [
    {{
      "action": "<BUY|SHORT|HOLD|WATCH>",
      "reasoning": "<specific reasoning citing article or macro condition>",
      "confidence": "<HIGH|MEDIUM|LOW>",
      "time_horizon": "<INTRADAY|SWING (1-5 days)|MEDIUM (weeks)|LONG (months)>",
      "source_article": "<title of the article that informed this signal, or 'Macro inference' if none>"
    }}
  ],
  "bull_case": "<what would make this ticker go up>",
  "bear_case": "<what would make this ticker go down>",
  "key_risk": "<single biggest risk right now>",
  "affected_sectors": ["<sector>"],
  "direct_articles_found": <true if any direct articles found, false if macro only>
}}

Return ONLY a valid JSON array. No markdown, no backticks, no explanation.

News articles:
{article_text}"""

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=4000,
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
    except json.JSONDecodeError:
        st.error("❌ AI returned malformed response. Try again.")
        return []
    except Exception as e:
        st.error(f"❌ Groq error: {str(e)[:150]}")
        return []


def _render_ticker_card(result: dict, relevant_articles: list):
    """Render a full analysis card for one ticker."""

    ticker          = result.get("ticker", "?")
    company_name    = result.get("company_name", "")
    overall_signal  = result.get("overall_signal", "WATCH")
    overall_conf    = result.get("overall_confidence", "LOW")
    market_impact   = result.get("market_impact", "NEUTRAL")
    urgency         = result.get("urgency", "LOW")
    summary         = result.get("summary", "")
    signals         = result.get("signals", [])
    bull_case       = result.get("bull_case", "")
    bear_case       = result.get("bear_case", "")
    key_risk        = result.get("key_risk", "")
    sectors         = ", ".join(result.get("affected_sectors", []))
    direct_found    = result.get("direct_articles_found", False)

    # Colour maps
    impact_icon  = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(market_impact, "⚪")
    urgency_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}.get(urgency, "")
    signal_emoji = {"BUY": "📈", "SHORT": "📉", "HOLD": "➡️", "WATCH": "👁️"}
    conf_icon    = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔘"}.get(overall_conf, "🔘")
    action_dot   = {"BUY": "🟢", "SHORT": "🔴", "HOLD": "🟡", "WATCH": "🔵"}

    st.markdown("---")

    # ── Title bar ──────────────────────────────────────────────────────────
    t1, t2, t3 = st.columns([3, 2, 2])
    with t1:
        st.markdown(f"## `{ticker}`")
        if company_name:
            st.caption(company_name)
    with t2:
        st.markdown(f"**{impact_icon} {market_impact}** &nbsp; {urgency_icon} {urgency} urgency")
        st.markdown(f"**{signal_emoji.get(overall_signal, '👁️')} {overall_signal}** &nbsp; {conf_icon} {overall_conf} confidence")
    with t3:
        if not direct_found:
            st.warning("⚠️ No direct news found — signals based on macro context only")
        else:
            st.success("✅ Direct news articles found")

    # ── Summary ────────────────────────────────────────────────────────────
    st.markdown(f"> {summary}")

    # ── Individual signals ─────────────────────────────────────────────────
    if signals:
        st.markdown("**📋 Signals**")
        for sig in signals:
            action      = sig.get("action", "WATCH")
            reasoning   = sig.get("reasoning", "")
            confidence  = sig.get("confidence", "LOW")
            horizon     = sig.get("time_horizon", "")
            source_art  = sig.get("source_article", "")
            dot         = action_dot.get(action, "⚪")
            emoji       = signal_emoji.get(action, "👁️")
            conf_dot    = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔘"}.get(confidence, "🔘")

            with st.container():
                sc1, sc2 = st.columns([1, 5])
                with sc1:
                    st.markdown(f"**{dot} {action}** {emoji}")
                    st.caption(f"{conf_dot} {confidence}")
                with sc2:
                    st.markdown(reasoning)
                    st.caption(f"⏱ {horizon}  ·  📰 {source_art}")

    # ── Bull / Bear / Risk ─────────────────────────────────────────────────
    st.markdown("")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.markdown("**🟢 Bull Case**")
        st.caption(bull_case)
    with bc2:
        st.markdown("**🔴 Bear Case**")
        st.caption(bear_case)
    with bc3:
        st.markdown("**⚠️ Key Risk**")
        st.caption(key_risk)

    if sectors:
        st.caption(f"📂 Sectors: {sectors}")

    # ── Relevant articles used ─────────────────────────────────────────────
    direct_arts = [a for a in relevant_articles if a.get("match_type") == "direct"]
    if direct_arts:
        with st.expander(f"📰 {len(direct_arts)} relevant article(s) found"):
            for art in direct_arts:
                st.markdown(f"- [{art['title']}]({art['url']}) — *{art['source']}*")


def run_ticker_analysis(api_key: str):

    st.markdown("**Enter any ticker(s) to get an AI-powered signal based on live news:**")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        ticker_input = st.text_input(
            "Ticker(s)",
            placeholder="e.g. TSLA, NVDA, BTC-USD",
            label_visibility="collapsed",
            key="ticker_analysis_input"
        )
    with col_btn:
        run_btn = st.button("🔍 Analyse", key="run_ticker_analysis", use_container_width=True)

    # ── Example chips ──────────────────────────────────────────────────────
    st.caption("Examples:")
    chip_cols = st.columns(6)
    examples = ["TSLA", "NVDA", "BTC-USD", "AAPL", "XLE", "SPY"]
    for i, ex in enumerate(examples):
        with chip_cols[i]:
            if st.button(ex, key=f"chip_{ex}", use_container_width=True):
                ticker_input = ex
                run_btn = True

    if not run_btn or not ticker_input.strip():
        # Show previous results if any
        if st.session_state.get("ticker_analysis_results"):
            st.markdown("---")
            st.caption("⬇️ Previous analysis — enter new tickers to refresh")
            _display_results(
                st.session_state["ticker_analysis_results"],
                st.session_state.get("ticker_analysis_articles", [])
            )
        return

    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not tickers:
        st.error("❌ Please enter at least one ticker.")
        return

    # ── Fetch articles ─────────────────────────────────────────────────────
    with st.spinner("📡 Scanning live news sources…"):
        all_articles = _fetch_all_articles()

    relevant = _filter_relevant(all_articles, tickers)

    direct_count = sum(1 for a in relevant if a.get("match_type") == "direct")
    macro_count  = sum(1 for a in relevant if a.get("match_type") == "macro")

    st.caption(f"Found {direct_count} direct article(s) + {macro_count} macro context article(s) across {len(RSS_FEEDS)} sources")

    # ── Analyse with Groq ──────────────────────────────────────────────────
    with st.spinner(f"🧠 AI analysing {', '.join(tickers)}…"):
        results = _analyze_tickers_with_groq(tickers, relevant, api_key)

    if not results:
        return

    # Store in session
    st.session_state["ticker_analysis_results"]  = results
    st.session_state["ticker_analysis_articles"] = relevant
    st.session_state["ticker_analysis_time"]     = datetime.now().strftime("%H:%M:%S")

    _display_results(results, relevant)


def _display_results(results: list, relevant_articles: list):
    time_str = st.session_state.get("ticker_analysis_time", "")
    if time_str:
        st.caption(f"Analysis generated at {time_str} — refresh to update")

    for result in results:
        _render_ticker_card(result, relevant_articles)

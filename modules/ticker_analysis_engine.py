"""
Ticker Analysis Engine
- User enters specific ticker(s)
- First checks already-processed Live Intelligence articles in session state
- Then fetches fresh from RSS feeds
- Groq analyses everything and returns per-ticker signals
"""

import streamlit as st
import feedparser
import re
import json
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

# Broad alias list — errs on side of inclusion
TICKER_ALIASES = {
    "TSLA":    ["tesla", "elon musk", "tsla", "cybertruck", "model s", "model 3", "model y", "model x", "supercharger"],
    "NVDA":    ["nvidia", "nvda", "jensen huang", "geforce", "cuda", "h100", "blackwell", "rtx"],
    "AAPL":    ["apple", "aapl", "tim cook", "iphone", "ipad", "macbook", "app store", "ios", "vision pro"],
    "MSFT":    ["microsoft", "msft", "satya nadella", "azure", "copilot", "xbox", "teams", "windows", "openai"],
    "AMZN":    ["amazon", "amzn", "aws", "andy jassy", "prime", "alexa", "whole foods"],
    "GOOGL":   ["google", "googl", "alphabet", "sundar pichai", "gemini", "youtube", "waymo", "chrome"],
    "GOOG":    ["google", "alphabet", "sundar pichai", "gemini", "youtube"],
    "META":    ["meta", "facebook", "instagram", "zuckerberg", "whatsapp", "threads", "ray-ban", "llama"],
    "AMD":     ["amd", "advanced micro devices", "lisa su", "ryzen", "radeon", "epyc", "instinct"],
    "INTC":    ["intel", "intc", "pat gelsinger", "arc", "xeon"],
    "NFLX":    ["netflix", "nflx", "reed hastings"],
    "DIS":     ["disney", "dis", "bob iger", "marvel", "star wars", "pixar", "hulu", "espn"],
    "BABA":    ["alibaba", "baba", "taobao", "tmall", "alipay", "jack ma"],
    "PDD":     ["pdd", "temu", "pinduoduo", "chen lei"],
    "TSM":     ["tsmc", "tsm", "taiwan semiconductor", "cc wei"],
    "BRKB":    ["berkshire", "warren buffett", "charlie munger", "brkb", "brka"],
    "JPM":     ["jpmorgan", "jpm", "jamie dimon", "chase"],
    "GS":      ["goldman sachs", "goldman", "gs", "david solomon"],
    "BAC":     ["bank of america", "bac", "brian moynihan"],
    "WMT":     ["walmart", "wmt", "doug mcmillon"],
    "UNH":     ["unitedhealth", "unh", "andrew witty"],
    "XOM":     ["exxon", "xom", "exxonmobil", "darren woods"],
    "CVX":     ["chevron", "cvx", "mike wirth"],
    "BTC-USD": ["bitcoin", "btc", "satoshi", "crypto", "cryptocurrency", "blockchain", "halving", "coinbase"],
    "ETH-USD": ["ethereum", "eth", "vitalik", "crypto", "defi", "web3", "smart contract"],
    "XLE":     ["energy", "oil", "crude", "petroleum", "exxon", "chevron", "natural gas", "opec",
                "barrel", "brent", "wti", "refinery", "drilling", "shale", "pipeline",
                "hormuz", "strait", "energy sector", "energy etf"],
    "XLF":     ["banking", "finance", "jpmorgan", "goldman", "bank of america", "financial sector",
                "federal reserve", "interest rate", "fed rate", "banks", "lending"],
    "XLK":     ["tech sector", "technology etf", "semiconductors", "software"],
    "XLV":     ["healthcare", "pharma", "biotech", "health sector"],
    "XLI":     ["industrial", "defense", "aerospace", "manufacturing"],
    "XLP":     ["consumer staples", "grocery", "household", "procter", "colgate"],
    "XLY":     ["consumer discretionary", "retail", "amazon", "tesla", "consumer spending"],
    "QQQ":     ["nasdaq", "qqq", "tech stocks", "technology index", "nasdaq 100"],
    "SPY":     ["s&p 500", "s&p500", "sp500", "spx", "spy", "market rally", "market crash",
                "stock market", "equities", "index", "wall street"],
    "GLD":     ["gold", "precious metals", "bullion", "safe haven", "gld"],
    "SLV":     ["silver", "slv", "precious metals"],
    "TLT":     ["treasury", "bonds", "tlt", "10-year", "yield", "interest rate", "fixed income"],
    "VIX":     ["volatility", "vix", "fear index", "options", "market fear"],
    "OIL":     ["oil", "crude", "petroleum", "brent", "wti", "opec", "barrel"],
    "^GSPC":   ["s&p 500", "s&p500", "sp500", "spx", "market rally", "stock market", "equities"],
    "^DJI":    ["dow jones", "dow", "djia", "blue chip"],
    "^IXIC":   ["nasdaq", "nasdaq composite", "tech stocks"],
    "^VIX":    ["volatility", "vix", "fear index", "market fear"],
}


def _get_aliases(ticker: str) -> list:
    ticker_upper = ticker.upper()
    base = TICKER_ALIASES.get(ticker_upper, [])
    # Always include the raw ticker itself
    ticker_lower = ticker.lower().replace("-usd", "").replace("^", "")
    if ticker_lower not in base:
        base = [ticker_lower] + base
    return base


def _harvest_live_intelligence_articles() -> list:
    """
    Pull articles already processed by the Live Intelligence tab
    from session state. These are the freshest, already-seen articles.
    """
    processed = st.session_state.get("processed_signals", [])
    harvested = []
    for article, analysis in processed:
        harvested.append({
            "source":     article.get("source", "Live Feed"),
            "title":      article.get("title", ""),
            "summary":    article.get("summary", ""),
            "url":        article.get("url", ""),
            "full_text":  (article.get("title", "") + " " + article.get("summary", "")).lower(),
            "match_type": "live_feed",
            # Include the AI analysis already done on this article
            "live_analysis": analysis,
        })
    return harvested


def _fetch_fresh_articles() -> list:
    """Fetch fresh articles from RSS feeds."""
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
                    "source":    source,
                    "title":     title,
                    "summary":   summary[:600],
                    "url":       url,
                    "full_text": (title + " " + summary).lower(),
                    "match_type": "fresh_feed",
                })
        except Exception:
            continue
    return articles


def _filter_relevant(articles: list, tickers: list) -> list:
    """
    Filter articles relevant to the requested tickers.
    Checks both direct ticker/alias matches AND macro context.
    Deduplicates by URL.
    """
    relevant = []
    seen_urls = set()

    # Build combined alias list for all requested tickers
    all_terms = []
    for t in tickers:
        all_terms.extend(_get_aliases(t))
    all_terms = list(set(all_terms))  # deduplicate

    # Pass 1: direct matches
    for art in articles:
        if art["url"] in seen_urls:
            continue
        text = art["full_text"]
        if any(term in text for term in all_terms):
            art_copy = dict(art)
            art_copy["match_type"] = "direct"
            relevant.append(art_copy)
            seen_urls.add(art["url"])

    # Pass 2: macro context (up to 6 extras)
    macro_terms = ["s&p", "nasdaq", "market", "fed", "inflation", "interest rate",
                   "economy", "gdp", "recession", "oil", "tariff", "trade",
                   "earnings", "unemployment", "cpi", "fomc", "treasury", "yield",
                   "geopolitical", "war", "sanctions", "opec", "dollar", "currency"]
    for art in articles:
        if art["url"] in seen_urls:
            continue
        if len([r for r in relevant if r.get("match_type") == "macro"]) >= 6:
            break
        text = art["full_text"]
        if any(term in text for term in macro_terms):
            art_copy = dict(art)
            art_copy["match_type"] = "macro"
            relevant.append(art_copy)
            seen_urls.add(art["url"])

    return relevant


def _analyze_tickers_with_groq(tickers: list, articles: list, api_key: str) -> list:
    import time
    from groq import Groq, RateLimitError, APIStatusError, APIConnectionError

    ticker_list = ", ".join(tickers)

    # Build article context — flag live feed articles specially
    article_text = ""
    for i, art in enumerate(articles[:14], 1):
        match_label = {
            "direct":     "DIRECT MATCH",
            "live_feed":  "LIVE FEED (already AI-processed)",
            "macro":      "MACRO CONTEXT",
        }.get(art.get("match_type", "macro"), "CONTEXT")

        article_text += f"\n[{i}] [{match_label}] {art['source']}\nTITLE: {art['title']}\nSUMMARY: {art['summary']}\nURL: {art['url']}\n"

        # If this article was already analysed by live intelligence, include that analysis
        if art.get("live_analysis"):
            la = art["live_analysis"]
            article_text += f"PRIOR AI ANALYSIS: market_impact={la.get('market_impact','?')}, urgency={la.get('urgency','?')}, reasoning={la.get('impact_reasoning','')[:200]}\n"

    prompt = f"""You are a professional Wall Street analyst. The user wants a full signal analysis for: {ticker_list}

Below are recent news articles — some directly mention the ticker(s), some are from the live feed (already AI-processed), and some provide macro context.

CONFIDENCE RULES — follow these strictly:
- HIGH: Direct news explicitly about this ticker from today
- MEDIUM: Related news, sector news, or reasonable inference from macro
- LOW: Only macro context available, no direct or sector news
- Never fake confidence. LOW with a real signal is better than inflated HIGH
- If live feed articles are included, weight them heavily — they are real and fresh

For EACH ticker in [{ticker_list}], return a JSON object in an array:
{{
  "ticker": "<ticker>",
  "company_name": "<full name>",
  "overall_signal": "<BUY|SHORT|HOLD|WATCH>",
  "overall_confidence": "<HIGH|MEDIUM|LOW>",
  "market_impact": "<BULLISH|BEARISH|NEUTRAL>",
  "urgency": "<HIGH|MEDIUM|LOW>",
  "summary": "<2-3 sentence honest assessment right now>",
  "signals": [
    {{
      "action": "<BUY|SHORT|HOLD|WATCH>",
      "reasoning": "<cite the specific article title or macro condition>",
      "confidence": "<HIGH|MEDIUM|LOW>",
      "time_horizon": "<INTRADAY|SWING (1-5 days)|MEDIUM (weeks)|LONG (months)>",
      "source_article": "<article title or 'Macro inference'>"
    }}
  ],
  "bull_case": "<specific bull scenario>",
  "bear_case": "<specific bear scenario>",
  "key_risk": "<biggest risk right now>",
  "affected_sectors": ["<sector>"],
  "direct_articles_found": <true|false>
}}

Return ONLY a valid JSON array. No markdown, no backticks.

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

        except RateLimitError:
            if attempt < 3:
                st.toast(f"⏳ Signal engine busy — retrying in {wait}s ({attempt}/3)", icon="⚡")
                time.sleep(wait)
                wait *= 2
            else:
                st.warning(
                    "⚡ **Rate limit reached.** Too many requests right now. "
                    "Wait 60 seconds and try again."
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
                    st.error(f"🔴 **Ticker signal engine error (HTTP {code})** — {str(e)[:120]}")
                    return []

        except json.JSONDecodeError:
            st.error("❌ Signal engine returned a malformed response. Try again.")
            return []

        except Exception as e:
            if attempt < 3:
                time.sleep(wait); wait *= 2
            else:
                st.error(f"❌ Ticker signal error: {str(e)[:150]}")
                return []

    return []


def _render_ticker_card(result: dict, relevant_articles: list):
    ticker         = result.get("ticker", "?")
    company_name   = result.get("company_name", "")
    overall_signal = result.get("overall_signal", "WATCH")
    overall_conf   = result.get("overall_confidence", "LOW")
    market_impact  = result.get("market_impact", "NEUTRAL")
    urgency        = result.get("urgency", "LOW")
    summary        = result.get("summary", "")
    signals        = result.get("signals", [])
    bull_case      = result.get("bull_case", "")
    bear_case      = result.get("bear_case", "")
    key_risk       = result.get("key_risk", "")
    sectors        = ", ".join(result.get("affected_sectors", []))
    direct_found   = result.get("direct_articles_found", False)

    impact_icon  = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(market_impact, "⚪")
    urgency_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}.get(urgency, "")
    signal_emoji = {"BUY": "📈", "SHORT": "📉", "HOLD": "➡️", "WATCH": "👁️"}
    conf_icon    = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔘"}.get(overall_conf, "🔘")
    action_dot   = {"BUY": "🟢", "SHORT": "🔴", "HOLD": "🟡", "WATCH": "🔵"}

    st.markdown("---")

    t1, t2, t3 = st.columns([3, 2, 2])
    with t1:
        st.markdown(f"## `{ticker}`")
        if company_name:
            st.caption(company_name)
    with t2:
        st.markdown(f"**{impact_icon} {market_impact}** &nbsp; {urgency_icon} {urgency} urgency")
        st.markdown(f"**{signal_emoji.get(overall_signal,'👁️')} {overall_signal}** &nbsp; {conf_icon} {overall_conf} confidence")
    with t3:
        # Count source types
        live_arts   = [a for a in relevant_articles if a.get("match_type") == "live_feed"]
        direct_arts = [a for a in relevant_articles if a.get("match_type") == "direct"]
        if live_arts:
            st.success(f"✅ {len(live_arts)} article(s) from Live Feed")
        if direct_arts:
            st.success(f"✅ {len(direct_arts)} direct article(s) found")
        if not live_arts and not direct_arts:
            st.warning("⚠️ No direct news — macro context only")

    st.markdown(f"> {summary}")

    if signals:
        st.markdown("**📋 Signals**")
        for sig in signals:
            action     = sig.get("action", "WATCH")
            reasoning  = sig.get("reasoning", "")
            confidence = sig.get("confidence", "LOW")
            horizon    = sig.get("time_horizon", "")
            source_art = sig.get("source_article", "")
            dot        = action_dot.get(action, "⚪")
            emoji      = signal_emoji.get(action, "👁️")
            conf_dot   = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔘"}.get(confidence, "🔘")

            sc1, sc2 = st.columns([1, 5])
            with sc1:
                st.markdown(f"**{dot} {action}** {emoji}")
                st.caption(f"{conf_dot} {confidence}")
            with sc2:
                st.markdown(reasoning)
                st.caption(f"⏱ {horizon}  ·  📰 {source_art}")

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

    # Show which articles were used
    all_used = [a for a in relevant_articles if a.get("match_type") in ("direct", "live_feed")]
    if all_used:
        with st.expander(f"📰 {len(all_used)} article(s) used in this analysis"):
            for art in all_used:
                source_tag = "🔴 LIVE" if art.get("match_type") == "live_feed" else "📡 DIRECT"
                st.markdown(f"- {source_tag} [{art['title']}]({art['url']}) — *{art['source']}*")


def run_ticker_analysis(api_key: str):

    st.markdown("**Enter any ticker(s) to get an AI signal based on live news:**")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        ticker_input = st.text_input(
            "Ticker(s)",
            placeholder="e.g. TSLA, NVDA, BTC-USD, XLE",
            label_visibility="collapsed",
            key="ticker_analysis_input"
        )
    with col_btn:
        run_btn = st.button("🔍 Analyse", key="run_ticker_analysis", use_container_width=True)

    st.caption("Quick examples:")
    chip_cols = st.columns(7)
    examples = ["TSLA", "NVDA", "BTC-USD", "AAPL", "XLE", "SPY", "META"]
    clicked_example = None
    for i, ex in enumerate(examples):
        with chip_cols[i]:
            if st.button(ex, key=f"chip_{ex}", use_container_width=True):
                clicked_example = ex

    if clicked_example:
        ticker_input = clicked_example
        run_btn = True

    # Show live feed status hint
    live_signals = st.session_state.get("processed_signals", [])
    if live_signals:
        st.info(f"📡 {len(live_signals)} article(s) already processed by Live Intelligence tab will be included automatically.")
    else:
        st.caption("💡 Tip: Run the Live Intelligence tab first to improve signal quality here.")

    if not run_btn or not ticker_input.strip():
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

    # ── Step 1: harvest Live Intelligence articles from session ────────────
    live_articles = _harvest_live_intelligence_articles()

    # ── Step 2: fetch fresh from RSS feeds ────────────────────────────────
    with st.spinner("📡 Scanning live news sources…"):
        fresh_articles = _fetch_fresh_articles()

    # ── Step 3: combine and filter ────────────────────────────────────────
    all_articles = live_articles + fresh_articles   # live feed articles first = higher priority
    relevant = _filter_relevant(all_articles, tickers)

    direct_count   = sum(1 for a in relevant if a.get("match_type") == "direct")
    live_count     = sum(1 for a in relevant if a.get("match_type") == "live_feed")
    macro_count    = sum(1 for a in relevant if a.get("match_type") == "macro")

    source_summary = []
    if live_count:   source_summary.append(f"{live_count} from Live Feed")
    if direct_count: source_summary.append(f"{direct_count} direct")
    if macro_count:  source_summary.append(f"{macro_count} macro context")
    st.caption(f"Articles used: {' · '.join(source_summary) if source_summary else 'macro context only'}")

    # ── Step 4: Groq analysis ─────────────────────────────────────────────
    with st.spinner(f"🧠 AI analysing {', '.join(tickers)}…"):
        results = _analyze_tickers_with_groq(tickers, relevant, api_key)

    if not results:
        return

    st.session_state["ticker_analysis_results"]  = results
    st.session_state["ticker_analysis_articles"] = relevant
    st.session_state["ticker_analysis_time"]     = datetime.now().strftime("%H:%M:%S")

    # Log all signals to history
    try:
        from modules.signal_history import log_signals_from_ticker
        for r in results:
            log_signals_from_ticker(r)
    except Exception:
        pass

    _display_results(results, relevant)


def _display_results(results: list, relevant_articles: list):
    time_str = st.session_state.get("ticker_analysis_time", "")
    if time_str:
        st.caption(f"Analysis generated at {time_str}")
    for result in results:
        _render_ticker_card(result, relevant_articles)

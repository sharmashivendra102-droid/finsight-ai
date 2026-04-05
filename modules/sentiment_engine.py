"""
News Sentiment Engine
- Scrapes article text via newspaper3k / BeautifulSoup fallback
- Scores each article with Groq LLM across 5 dimensions
- Saves results to CSV
"""

import streamlit as st
import pandas as pd
import json
import re
import time

import warnings
warnings.filterwarnings("ignore")

MIN_ARTICLE_LENGTH = 150  # characters


def _scrape_article(url: str) -> tuple[str, str]:
    """
    Returns (text, publish_date_str).
    Tries newspaper3k first, then BeautifulSoup fallback.
    """
    # ── Attempt 1: newspaper3k ─────────────────────────────────────────────
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        text = art.text.strip()
        pub_date = str(art.publish_date.date()) if art.publish_date else "unknown"
        if len(text) >= MIN_ARTICLE_LENGTH:
            return text, pub_date
    except Exception:
        pass

    # ── Attempt 2: BeautifulSoup fallback ─────────────────────────────────
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try article tag first, fall back to body
        article_tag = soup.find("article") or soup.find("main") or soup.body
        text = article_tag.get_text(separator=" ", strip=True) if article_tag else ""
        text = re.sub(r"\s+", " ", text).strip()

        # Try to find date
        time_tag = soup.find("time")
        pub_date = time_tag.get("datetime", "unknown")[:10] if time_tag else "unknown"

        if len(text) >= MIN_ARTICLE_LENGTH:
            return text, pub_date
        return "", "unknown"

    except Exception:
        return "", "unknown"


def _score_with_groq(text: str, url: str, api_key: str) -> dict:
    """
    Calls Groq LLM to score the article.
    Returns dict with keys: political_bias, truthfulness, propaganda, hype, panic
    All values clamped 0-1.
    """
    from groq import Groq

    # Truncate text to keep prompt within token limits
    snippet = text[:3500]

    prompt = f"""You are a financial journalism analyst. Analyze the following news article excerpt and return ONLY a JSON object with these exact keys and float values between 0 and 1:
- political_bias: 0 = no bias, 1 = extreme political bias
- truthfulness: 0 = very untruthful/speculative, 1 = highly factual and verified
- propaganda: 0 = no propaganda, 1 = clear propaganda
- hype: 0 = measured tone, 1 = extreme hype/exaggeration
- panic: 0 = calm reporting, 1 = extreme fear/panic-inducing

Article URL: {url}

Article excerpt:
\"\"\"
{snippet}
\"\"\"

Respond with ONLY the JSON object. No markdown, no explanation, no backticks. Example:
{{"political_bias": 0.2, "truthfulness": 0.8, "propaganda": 0.1, "hype": 0.3, "panic": 0.1}}"""

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Extract JSON object
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        scores = json.loads(raw)

        # Validate and clamp
        required_keys = ["political_bias", "truthfulness", "propaganda", "hype", "panic"]
        result = {}
        for k in required_keys:
            val = scores.get(k, 0.5)
            try:
                result[k] = float(max(0.0, min(1.0, val)))
            except (TypeError, ValueError):
                result[k] = 0.5

        return result

    except json.JSONDecodeError:
        st.warning(f"⚠️ LLM returned malformed JSON for one article; using neutral defaults.")
        return {k: 0.5 for k in ["political_bias", "truthfulness", "propaganda", "hype", "panic"]}
    except Exception as e:
        st.warning(f"⚠️ Groq API error: {str(e)[:120]}")
        return {k: 0.5 for k in ["political_bias", "truthfulness", "propaganda", "hype", "panic"]}


def run_sentiment_analysis(urls: list, groq_api_key: str):
    results = []
    prog = st.progress(0, text="Analysing articles…")

    for i, url in enumerate(urls):
        prog.progress((i) / len(urls), text=f"Processing {i+1}/{len(urls)}: {url[:60]}…")

        # Scrape
        text, pub_date = _scrape_article(url)

        if not text:
            st.warning(f"⚠️ Could not extract content from: `{url}`")
            continue

        st.caption(f"📄 Scraped {len(text):,} chars from {url[:70]}…")

        # Score
        scores = _score_with_groq(text, url, groq_api_key)

        results.append({
            "url": url,
            "publish_date": pub_date,
            "characters": len(text),
            **scores
        })

        # Small delay to be polite to APIs
        time.sleep(0.5)

    prog.empty()

    if not results:
        st.error("❌ No articles could be processed. Check the URLs and try again.")
        return

    sentiment_df = pd.DataFrame(results)
    sentiment_df.to_csv("news_sentiment_scores.csv", index=False)

    st.success(f"✅ Scored {len(sentiment_df)} article(s). Saved to `news_sentiment_scores.csv`")

    # ── Display table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sentiment Scores</div>', unsafe_allow_html=True)

    score_cols = ["political_bias", "truthfulness", "propaganda", "hype", "panic"]

    def _fmt_score(val):
        if val >= 0.7:
            color = "#f87171"  # red — bad
        elif val >= 0.4:
            color = "#fbbf24"  # amber — medium
        else:
            color = "#4ade80"  # green — good
        # Invert color logic for truthfulness (higher is better)
        return f"color: {color}"

    def _fmt_truth(val):
        if val >= 0.7:
            return "color: #4ade80"
        elif val >= 0.4:
            return "color: #fbbf24"
        else:
            return "color: #f87171"

    display_df = sentiment_df[["url", "publish_date"] + score_cols].copy()

    styler = display_df.style
    for col in ["political_bias", "propaganda", "hype", "panic"]:
        styler = styler.map(_fmt_score, subset=[col])
    styler = styler.map(_fmt_truth, subset=["truthfulness"])
    styler = styler.format({c: "{:.2f}" for c in score_cols})

    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Averages ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Average Scores Across All Articles</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    labels = {
        "political_bias": ("Political Bias", "🏛️"),
        "truthfulness":   ("Truthfulness",   "✅"),
        "propaganda":     ("Propaganda",     "📢"),
        "hype":           ("Hype",           "🔥"),
        "panic":          ("Panic",          "😱"),
    }

    for i, (key, (label, icon)) in enumerate(labels.items()):
        avg = sentiment_df[key].mean()
        with cols[i]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="label">{icon} {label}</div>
                <div class="value">{avg:.2f}</div>
            </div>""", unsafe_allow_html=True)

    # ── Bar chart ──────────────────────────────────────────────────────────
    if len(sentiment_df) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        CARD   = "#0d1b2a"
        BORDER = "#1e3a5f"
        MUTED  = "#6b8fad"
        CYAN   = "#7dd3fc"
        TEXT   = "#c9d8e8"

        avg_vals = [sentiment_df[k].mean() for k in score_cols]
        colors = ["#60a5fa", "#4ade80", "#f87171", "#fbbf24", "#f472b6"]

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        bars = ax.bar(score_cols, avg_vals, color=colors, edgecolor=BORDER, linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_title("Average Sentiment Scores", color=CYAN, fontsize=11)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(axis="y", color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)

        for bar, val in zip(bars, avg_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", color=TEXT, fontsize=9)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Save to session for Portfolio Risk tab
    st.session_state["sentiment_df"] = sentiment_df

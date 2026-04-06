"""
AI Advisor Engine
- Synthesizes correlation data, sentiment scores, and risk flags
- Sends structured context to Groq LLM
- Returns actionable, specific financial advice
"""

import streamlit as st
import json
import re


def _build_context(corr_df, sentiment_df, roll_df, valid_tickers) -> str:
    """Build a rich text context block from all available data."""
    lines = []

    # ── Tickers ────────────────────────────────────────────────────────────
    if valid_tickers:
        lines.append(f"PORTFOLIO TICKERS: {', '.join(valid_tickers)}")
        lines.append("")

    # ── Correlations ───────────────────────────────────────────────────────
    if corr_df is not None and not corr_df.empty:
        lines.append("PAIRWISE CORRELATIONS (full period):")
        for _, row in corr_df.iterrows():
            lines.append(f"  {row['Ticker 1']} / {row['Ticker 2']}: {row['Correlation']:.4f}")
        lines.append("")

        # Summarise clusters
        high  = corr_df[corr_df["Correlation"] >= 0.75]
        neg   = corr_df[corr_df["Correlation"] <= -0.4]
        if not high.empty:
            pairs = ", ".join(f"{r['Ticker 1']}/{r['Ticker 2']} ({r['Correlation']:.2f})" for _, r in high.iterrows())
            lines.append(f"HIGH CORRELATION CLUSTERS (≥0.75): {pairs}")
        if not neg.empty:
            pairs = ", ".join(f"{r['Ticker 1']}/{r['Ticker 2']} ({r['Correlation']:.2f})" for _, r in neg.iterrows())
            lines.append(f"NEGATIVE CORRELATIONS (≤-0.4, potential hedges): {pairs}")
        lines.append("")

    # ── Recent rolling correlation trend ───────────────────────────────────
    if roll_df is not None and not roll_df.empty:
        recent = roll_df.tail(30)
        lines.append("RECENT 30-DAY ROLLING CORRELATION AVERAGES:")
        for col in recent.columns:
            avg = recent[col].mean()
            trend = recent[col].iloc[-1] - recent[col].iloc[0]
            direction = "↑ rising" if trend > 0.05 else ("↓ falling" if trend < -0.05 else "→ stable")
            lines.append(f"  {col.replace('_', ' / ')}: avg={avg:.3f}, trend={direction}")
        lines.append("")

    # ── Sentiment ──────────────────────────────────────────────────────────
    if sentiment_df is not None and not sentiment_df.empty:
        lines.append(f"NEWS SENTIMENT ANALYSIS ({len(sentiment_df)} articles):")
        score_cols = ["political_bias", "truthfulness", "propaganda", "hype", "panic"]
        for col in score_cols:
            avg = sentiment_df[col].mean()
            lines.append(f"  {col}: {avg:.2f}")
        lines.append("")

        # Per-article breakdown
        lines.append("PER-ARTICLE DETAIL:")
        for _, row in sentiment_df.iterrows():
            lines.append(f"  URL: {row['url'][:80]}")
            lines.append(f"    hype={row['hype']:.2f}, panic={row['panic']:.2f}, "
                         f"truthfulness={row['truthfulness']:.2f}, propaganda={row['propaganda']:.2f}")
        lines.append("")

    return "\n".join(lines)


def _call_groq_advisor(context: str, user_question: str, api_key: str, conversation_history: list) -> str:
    from groq import Groq

    system_prompt = """You are FinSight AI, an expert financial advisor and quantitative analyst. 
You have deep knowledge of portfolio theory, market correlations, technical analysis, and behavioral finance.

You have been given real market data including:
- Pairwise and rolling correlations between assets
- AI-scored news sentiment (hype, panic, truthfulness, propaganda, political bias)
- Portfolio risk signals

Your job is to give SPECIFIC, ACTIONABLE financial advice based on this data. 

Rules:
- Always ground advice in the actual numbers provided
- Be specific: name tickers, cite correlation values, reference sentiment scores
- Structure your response clearly with sections
- Give concrete recommendations (reduce exposure, consider hedging, etc.)
- Flag risks clearly
- Be direct — investors need clarity, not vagueness
- Always include a disclaimer at the end
- Do NOT make up data not provided to you
- If data is missing (e.g. no sentiment run yet), say so and advise based on what IS available

Format your advice with these sections when relevant:
📊 CORRELATION INSIGHTS
📰 SENTIMENT OUTLOOK  
⚠️ RISK ASSESSMENT
💡 SPECIFIC RECOMMENDATIONS
🔮 OUTLOOK
⚠️ DISCLAIMER"""

    messages = [{"role": "system", "content": system_prompt}]

    # Inject data context as first user message if this is a fresh conversation
    if not conversation_history:
        messages.append({
            "role": "user",
            "content": f"Here is the current market data for my portfolio:\n\n{context}\n\nBased on this data, {user_question}"
        })
    else:
        # Rebuild full history, prepend context to very first user message
        for i, msg in enumerate(conversation_history):
            if i == 0 and msg["role"] == "user":
                messages.append({
                    "role": "user",
                    "content": f"Here is the current market data for my portfolio:\n\n{context}\n\n{msg['content']}"
                })
            else:
                messages.append(msg)
        messages.append({"role": "user", "content": user_question})

    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error calling AI advisor: {str(e)}"


def run_advisor(groq_api_key: str):
    corr_df       = st.session_state.get("corr_df")
    sentiment_df  = st.session_state.get("sentiment_df")
    roll_df       = st.session_state.get("roll_df")
    valid_tickers = st.session_state.get("valid_tickers", [])

    has_corr = corr_df is not None and not corr_df.empty
    has_sent = sentiment_df is not None and not sentiment_df.empty

    # ── Data status banner ─────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if has_corr:
            st.markdown(f"""
            <div class="ok-flag">
                ✅ Correlation data loaded — {len(valid_tickers)} tickers
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-flag">
                ⚠️ No correlation data yet — run Correlation Analytics tab first
            </div>""", unsafe_allow_html=True)
    with c2:
        if has_sent:
            st.markdown(f"""
            <div class="ok-flag">
                ✅ Sentiment data loaded — {len(sentiment_df)} articles
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-flag">
                ⚠️ No sentiment data yet — run News Sentiment tab first (optional)
            </div>""", unsafe_allow_html=True)

    if not has_corr and not has_sent:
        st.info("ℹ️ Run at least the **Correlation Analytics** tab first to give the AI advisor data to work with.")
        return

    st.markdown("---")

    # ── Quick-start prompt buttons ─────────────────────────────────────────
    st.markdown("**Quick questions — click one or type your own below:**")

    quick_prompts = [
        "Give me a full portfolio analysis and your top recommendations.",
        "Is my portfolio well diversified? What should I change?",
        "What are the biggest risks in my current portfolio right now?",
        "Based on the news sentiment, should I be bullish or bearish?",
        "Which assets should I reduce exposure to and why?",
        "Are there any natural hedges I should be using?",
    ]

    cols = st.columns(3)
    clicked_prompt = None
    for i, prompt in enumerate(quick_prompts):
        with cols[i % 3]:
            if st.button(prompt, key=f"quick_{i}", use_container_width=True):
                clicked_prompt = prompt

    st.markdown("---")

    # ── Conversation history ───────────────────────────────────────────────
    if "advisor_history" not in st.session_state:
        st.session_state["advisor_history"] = []

    # Build context once
    context = _build_context(corr_df, sentiment_df, roll_df, valid_tickers)

    # ── Display chat history ───────────────────────────────────────────────
    for msg in st.session_state["advisor_history"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="📈"):
                st.markdown(msg["content"])

    # ── Input ──────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask the AI advisor anything about your portfolio…")

    # Use quick prompt if clicked, otherwise use typed input
    final_input = clicked_prompt or user_input

    if final_input:
        if not groq_api_key:
            st.error("❌ Groq API key not configured.")
            return

        # Show user message
        with st.chat_message("user"):
            st.markdown(final_input)

        # Get AI response
        with st.chat_message("assistant", avatar="📈"):
            with st.spinner("FinSight AI is analysing your portfolio…"):
                response = _call_groq_advisor(
                    context=context,
                    user_question=final_input,
                    api_key=groq_api_key,
                    conversation_history=st.session_state["advisor_history"].copy()
                )
            st.markdown(response)

        # Save to history
        st.session_state["advisor_history"].append({"role": "user",      "content": final_input})
        st.session_state["advisor_history"].append({"role": "assistant",  "content": response})

    # ── Clear conversation button ──────────────────────────────────────────
    if st.session_state["advisor_history"]:
        if st.button("🗑️ Clear conversation", key="clear_advisor"):
            st.session_state["advisor_history"] = []
            st.rerun()

"""
Portfolio Risk Insights
- Combines correlation clusters and sentiment scores
- Generates risk flags
"""

import streamlit as st
import pandas as pd


def run_portfolio_risk():
    corr_df      = st.session_state.get("corr_df")
    sentiment_df = st.session_state.get("sentiment_df")
    valid_tickers = st.session_state.get("valid_tickers", [])

    has_corr = corr_df is not None and not corr_df.empty
    has_sent = sentiment_df is not None and not sentiment_df.empty

    if not has_corr and not has_sent:
        st.info("ℹ️ Run **Correlation Analytics** and/or **News Sentiment** first to generate risk signals.")
        return

    flags = []   # list of (level, message, detail)

    # ── Correlation-based signals ──────────────────────────────────────────
    if has_corr:
        st.markdown('<div class="section-header">📊 Correlation Risk Signals</div>', unsafe_allow_html=True)

        high_corr = corr_df[corr_df["Correlation"] >= 0.8]
        mid_corr  = corr_df[(corr_df["Correlation"] >= 0.5) & (corr_df["Correlation"] < 0.8)]
        neg_corr  = corr_df[corr_df["Correlation"] <= -0.5]

        if not high_corr.empty:
            pairs_str = ", ".join(
                f"{r['Ticker 1']}/{r['Ticker 2']} ({r['Correlation']:.2f})"
                for _, r in high_corr.iterrows()
            )
            flags.append(("risk", "🔴 High Correlation Cluster Detected",
                          f"Very high correlation (≥0.80): {pairs_str}. "
                          "These assets move almost in lockstep — your portfolio has reduced diversification."))

        if not mid_corr.empty:
            pairs_str = ", ".join(
                f"{r['Ticker 1']}/{r['Ticker 2']} ({r['Correlation']:.2f})"
                for _, r in mid_corr.iterrows()
            )
            flags.append(("warn", "🟡 Moderate Correlation Cluster",
                          f"Moderate correlation (0.50–0.79): {pairs_str}. "
                          "Some overlap in risk exposure; diversification is partial."))

        if not neg_corr.empty:
            pairs_str = ", ".join(
                f"{r['Ticker 1']}/{r['Ticker 2']} ({r['Correlation']:.2f})"
                for _, r in neg_corr.iterrows()
            )
            flags.append(("ok", "🟢 Natural Hedge Detected",
                          f"Negative correlation (≤-0.50): {pairs_str}. "
                          "These assets can act as natural hedges against each other."))

        if high_corr.empty and mid_corr.empty:
            flags.append(("ok", "🟢 Well-Diversified Correlation Profile",
                          "No pairs exceed 0.80 correlation — your selected assets show good diversification."))

        # Summary of diversification level
        avg_corr = corr_df["Correlation"].mean()
        st.markdown(f"""
        <div class="insight-card">
            <b style="color:#7dd3fc;">Correlation Overview</b><br>
            <span style="color:#c9d8e8;">
            Tickers analysed: <b>{', '.join(valid_tickers)}</b><br>
            Average pairwise correlation: <b>{avg_corr:.3f}</b><br>
            Highest correlation pair: <b>{corr_df.iloc[0]['Ticker 1']}</b> / <b>{corr_df.iloc[0]['Ticker 2']}</b>
            at <b>{corr_df.iloc[0]['Correlation']:.3f}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Sentiment-based signals ────────────────────────────────────────────
    if has_sent:
        st.markdown('<div class="section-header">📰 Sentiment Risk Signals</div>', unsafe_allow_html=True)

        avg_hype      = sentiment_df["hype"].mean()
        avg_panic     = sentiment_df["panic"].mean()
        avg_truth     = sentiment_df["truthfulness"].mean()
        avg_prop      = sentiment_df["propaganda"].mean()
        avg_bias      = sentiment_df["political_bias"].mean()
        n_articles    = len(sentiment_df)

        st.markdown(f"""
        <div class="insight-card">
            <b style="color:#7dd3fc;">Sentiment Overview</b><br>
            <span style="color:#c9d8e8;">
            Articles analysed: <b>{n_articles}</b><br>
            Avg Hype: <b>{avg_hype:.2f}</b> &nbsp;|&nbsp;
            Avg Panic: <b>{avg_panic:.2f}</b> &nbsp;|&nbsp;
            Avg Truthfulness: <b>{avg_truth:.2f}</b> &nbsp;|&nbsp;
            Avg Propaganda: <b>{avg_prop:.2f}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        if avg_hype >= 0.65 and avg_truth <= 0.5:
            flags.append(("risk", "🔴 High Hype + Low Truthfulness",
                          f"Average hype score {avg_hype:.2f} combined with low truthfulness {avg_truth:.2f} "
                          "suggests sensationalist coverage. Be cautious of speculative price moves driven by media."))

        elif avg_hype >= 0.65:
            flags.append(("warn", "🟡 Elevated Media Hype",
                          f"Average hype score {avg_hype:.2f} indicates exaggerated coverage. "
                          "This may cause short-term price volatility not supported by fundamentals."))

        if avg_panic >= 0.65:
            flags.append(("risk", "🔴 High Panic Signal",
                          f"Average panic score {avg_panic:.2f} — media is inducing fear. "
                          "Potential for volatility spikes, panic selling, or sharp price movements."))

        if avg_prop >= 0.6:
            flags.append(("warn", "🟡 Propaganda Detected in Coverage",
                          f"Average propaganda score {avg_prop:.2f}. "
                          "Some articles may be pushing a narrative. Cross-reference with primary sources."))

        if avg_bias >= 0.6:
            flags.append(("warn", "🟡 Political Bias in Coverage",
                          f"Average political bias {avg_bias:.2f}. "
                          "Coverage may be slanted — consider the editorial stance of your news sources."))

        if avg_hype < 0.4 and avg_panic < 0.4 and avg_truth >= 0.65:
            flags.append(("ok", "🟢 News Environment Looks Calm",
                          "Low hype, low panic, and reasonably high truthfulness suggest a rational news cycle."))

    # ── Combined signal ────────────────────────────────────────────────────
    if has_corr and has_sent:
        st.markdown('<div class="section-header">🔗 Combined Risk Signal</div>', unsafe_allow_html=True)

        avg_corr_val = corr_df["Correlation"].mean()
        avg_hype_val = sentiment_df["hype"].mean()
        avg_panic_val = sentiment_df["panic"].mean()

        risk_score = (avg_corr_val * 0.4) + (avg_hype_val * 0.3) + (avg_panic_val * 0.3)
        risk_score = max(0.0, min(1.0, risk_score))

        if risk_score >= 0.65:
            risk_label = "HIGH RISK"
            risk_color = "#f87171"
            risk_desc  = "Portfolio appears concentrated and news environment is heated. Consider reducing exposure."
        elif risk_score >= 0.40:
            risk_label = "MODERATE RISK"
            risk_color = "#fbbf24"
            risk_desc  = "Some correlation concentration and moderate media sentiment risk. Monitor closely."
        else:
            risk_label = "LOW RISK"
            risk_color = "#4ade80"
            risk_desc  = "Portfolio diversification and news sentiment look reasonable."

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#122338);border:1px solid {risk_color};
                    border-radius:14px;padding:1.6rem 2rem;margin-bottom:1rem;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:2rem;color:{risk_color};font-weight:700;">
                {risk_label}
            </div>
            <div style="color:#c9d8e8;font-size:1rem;margin-top:0.5rem;">
                Combined Risk Score: <b style="color:{risk_color};">{risk_score:.2f}</b>
            </div>
            <div style="color:#6b8fad;font-size:0.88rem;margin-top:0.4rem;">{risk_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Render all flags ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 All Risk Flags</div>', unsafe_allow_html=True)

    if not flags:
        st.info("No risk flags generated yet.")
    else:
        for level, title, detail in flags:
            css_class = "risk-flag" if level == "risk" else ("ok-flag" if level == "ok" else "insight-card")
            st.markdown(f"""
            <div class="{css_class}">
                <b>{title}</b><br>
                <span style="font-size:0.88rem;opacity:0.85;">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:2rem;padding:1rem;border:1px solid #1e3a5f;border-radius:10px;
                color:#6b8fad;font-size:0.8rem;">
        ⚠️ <b>Disclaimer:</b> FinSight AI is an analytical tool for informational purposes only.
        It does not constitute financial advice. Always conduct your own due diligence and consult
        a qualified financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)

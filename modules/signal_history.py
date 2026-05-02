"""
Signal History Log — Supabase backend (DEBUG VERSION)
=======================================================
This version shows ALL errors visibly instead of hiding them.

SQL to run in Supabase → SQL Editor:

    CREATE TABLE IF NOT EXISTS signals (
      id            BIGSERIAL PRIMARY KEY,
      timestamp     TEXT NOT NULL,
      source        TEXT NOT NULL,
      ticker        TEXT NOT NULL,
      action        TEXT NOT NULL,
      confidence    TEXT NOT NULL,
      urgency       TEXT,
      market_impact TEXT,
      time_horizon  TEXT,
      reasoning     TEXT,
      article_title TEXT,
      article_url   TEXT,
      source_feed   TEXT
    );

    CREATE TABLE IF NOT EXISTS holdings (
      id        BIGSERIAL PRIMARY KEY,
      ticker    TEXT NOT NULL UNIQUE,
      shares    REAL NOT NULL,
      avg_cost  REAL NOT NULL,
      notes     TEXT,
      added_at  TEXT
    );
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta


def _get_client():
    """
    Returns (client, error_string).
    client is None if anything failed.
    error_string is None if everything worked.
    Not cached so errors are always visible.
    """
    try:
        from supabase import create_client
    except ImportError as e:
        return None, f"supabase package not installed: {e}"

    try:
        url = st.secrets.get("SUPABASE_URL", "") or os.getenv("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "") or os.getenv("SUPABASE_KEY", "")
    except Exception as e:
        return None, f"Could not read secrets: {e}"

    if not url:
        return None, "SUPABASE_URL is missing from Streamlit Secrets"
    if not key:
        return None, "SUPABASE_KEY is missing from Streamlit Secrets"

    try:
        return create_client(url, key), None
    except Exception as e:
        return None, f"create_client failed: {e}"


def log_signal(
    source:        str,
    ticker:        str,
    action:        str,
    confidence:    str,
    urgency:       str  = "",
    market_impact: str  = "",
    time_horizon:  str  = "",
    reasoning:     str  = "",
    article_title: str  = "",
    article_url:   str  = "",
    source_feed:   str  = "",
):
    client, err = _get_client()
    if client is None:
        # Store error so the UI tab can surface it
        st.session_state["_supabase_error"] = err
        return

    try:
        client.table("signals").insert({
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source":        source,
            "ticker":        ticker.upper(),
            "action":        action.upper(),
            "confidence":    confidence.upper(),
            "urgency":       urgency        or "",
            "market_impact": market_impact  or "",
            "time_horizon":  time_horizon   or "",
            "reasoning":     (reasoning      or "")[:400],
            "article_title": (article_title  or "")[:200],
            "article_url":   (article_url    or "")[:500],
            "source_feed":   source_feed    or "",
        }).execute()
        st.session_state.pop("_supabase_error", None)
    except Exception as e:
        st.session_state["_supabase_error"] = f"INSERT failed: {e}"


def log_signals_from_live(article: dict, analysis: dict):
    for rec in analysis.get("recommendations", []):
        ticker = rec.get("ticker", "")
        if not ticker:
            continue
        log_signal(
            source        = "live_intelligence",
            ticker        = ticker,
            action        = rec.get("action",     "WATCH"),
            confidence    = rec.get("confidence", "LOW"),
            urgency       = analysis.get("urgency",       ""),
            market_impact = analysis.get("market_impact", ""),
            time_horizon  = rec.get("time_horizon", ""),
            reasoning     = rec.get("reasoning",    ""),
            article_title = article.get("title",    ""),
            article_url   = article.get("url",      ""),
            source_feed   = article.get("source",   ""),
        )


def log_signals_from_ticker(ticker_result: dict):
    ticker = ticker_result.get("ticker", "")
    for sig in ticker_result.get("signals", []):
        log_signal(
            source        = "ticker_signals",
            ticker        = ticker,
            action        = sig.get("action",        "WATCH"),
            confidence    = sig.get("confidence",    "LOW"),
            urgency       = ticker_result.get("urgency",       ""),
            market_impact = ticker_result.get("market_impact", ""),
            time_horizon  = sig.get("time_horizon",  ""),
            reasoning     = sig.get("reasoning",     ""),
            article_title = sig.get("source_article",""),
            article_url   = "",
            source_feed   = "ticker_analysis",
        )


def log_signals_from_briefing(tickers_to_watch: list):
    for tw in tickers_to_watch:
        log_signal(
            source        = "market_briefing",
            ticker        = tw.get("ticker",    ""),
            action        = tw.get("action",    "WATCH"),
            confidence    = tw.get("confidence","LOW"),
            urgency       = "MEDIUM",
            market_impact = "",
            time_horizon  = "",
            reasoning     = tw.get("reasoning", ""),
            article_title = tw.get("catalyst",  ""),
            article_url   = "",
            source_feed   = "market_briefing",
        )


def get_signals_df(
    days_back:     int  = 30,
    source_filter: list = None,
    action_filter: list = None,
) -> pd.DataFrame:
    client, err = _get_client()
    if client is None:
        return pd.DataFrame()
    try:
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M:%S")
        query = (
            client.table("signals")
            .select("*")
            .gte("timestamp", cutoff)
            .order("timestamp", desc=True)
        )
        if source_filter:
            query = query.in_("source", source_filter)
        if action_filter:
            query = query.in_("action", action_filter)
        resp = query.limit(5000).execute()
        rows = resp.data if resp.data else []
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        st.session_state["_supabase_error"] = f"SELECT failed: {e}"
        return pd.DataFrame()


def get_signal_count() -> int:
    client, err = _get_client()
    if client is None:
        return 0
    try:
        resp = client.table("signals").select("id", count="exact").execute()
        return resp.count or 0
    except Exception:
        return 0


def get_signal_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    total       = len(df)
    by_act      = df["action"].value_counts().to_dict()
    by_conf     = df["confidence"].value_counts().to_dict()
    by_src      = df["source"].value_counts().to_dict()
    top_tickers = df["ticker"].value_counts().head(10).to_dict()
    high_conf   = df[df["confidence"] == "HIGH"]
    return {
        "total":         total,
        "by_action":     by_act,
        "by_confidence": by_conf,
        "by_source":     by_src,
        "top_tickers":   top_tickers,
        "high_conf_pct": round(len(high_conf) / total * 100, 1) if total > 0 else 0,
        "date_range":    f"{df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}",
    }


def delete_all_signals():
    client, err = _get_client()
    if client is None:
        return
    try:
        client.table("signals").delete().gt("id", 0).execute()
    except Exception as e:
        st.error(f"Could not clear signals: {e}")


def run_signal_history():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    CARD   = "#0d1b2a"; BORDER = "#1e3a5f"; MUTED = "#6b8fad"
    CYAN   = "#7dd3fc"; TEXT   = "#c9d8e8"; GREEN = "#4ade80"
    RED    = "#f87171"; AMBER  = "#fbbf24"; BLUE  = "#38bdf8"

    # ── ALWAYS-VISIBLE DIAGNOSTICS ────────────────────────────────────────
    st.markdown('<div class="section-header">🔧 Supabase Connection Status</div>',
                unsafe_allow_html=True)

    client, conn_err = _get_client()

    if conn_err:
        st.error(f"❌ **Connection failed:** {conn_err}")
        if "not installed" in conn_err:
            st.info("Add `supabase>=2.0.0` to requirements.txt and push to GitHub.")
        elif "missing" in conn_err:
            st.info("Go to Streamlit Cloud → Settings → Secrets and add SUPABASE_URL and SUPABASE_KEY.")
        elif "create_client" in conn_err:
            st.info("Your URL or key is malformed. Copy fresh from Supabase → Project Settings → API.")
    else:
        st.success("✅ Supabase client connected")

        try:
            resp  = client.table("signals").select("id", count="exact").execute()
            count = resp.count or 0
            st.success(f"✅ signals table readable — **{count} rows**")
        except Exception as e:
            st.error(f"❌ Cannot read signals table: {e}")
            st.info("Run CREATE TABLE SQL in Supabase → SQL Editor")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Test Write → signals", key="test_write_sig"):
                try:
                    client.table("signals").insert({
                        "timestamp":  "2000-01-01 00:00:00",
                        "source":     "test",
                        "ticker":     "TEST",
                        "action":     "BUY",
                        "confidence": "LOW",
                    }).execute()
                    client.table("signals").delete().eq("ticker", "TEST").execute()
                    st.success("✅ Write test PASSED")
                except Exception as e:
                    st.error(f"❌ Write failed: {e}")
                    st.warning(
                        "**Most likely cause: Row Level Security (RLS).**\n\n"
                        "Supabase → Table Editor → signals table → "
                        "look for the RLS toggle at the top right → turn it OFF.\n\n"
                        "Do the same for holdings table."
                    )
        with col2:
            if st.button("🧪 Test Write → holdings", key="test_write_hold"):
                try:
                    client.table("holdings").upsert({
                        "ticker":   "TEST",
                        "shares":   1.0,
                        "avg_cost": 1.0,
                    }, on_conflict="ticker").execute()
                    client.table("holdings").delete().eq("ticker", "TEST").execute()
                    st.success("✅ Write test PASSED")
                except Exception as e:
                    st.error(f"❌ Write failed: {e}")
                    st.warning(
                        "**RLS is blocking writes.**\n"
                        "Supabase → Table Editor → holdings → toggle RLS off."
                    )

    # Show any background logging error
    bg_err = st.session_state.get("_supabase_error")
    if bg_err:
        st.error(f"❌ **Background logging error** (from Live Feed / Ticker Lookup):\n`{bg_err}`")

    st.markdown("---")

    # ── Normal UI ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Persistent signal log — every signal from every tab.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Saved to Supabase — survives redeploys permanently.
        </span>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        days_back = st.selectbox(
            "Time period", [1,7,14,30,90,365], index=3,
            format_func=lambda x: f"Last {x} day{'s' if x>1 else ''}"
        )
    with f2:
        source_filter = st.multiselect(
            "Source",
            ["live_intelligence","ticker_signals","market_briefing",
             "strategy_signals","auto_evaluator"],
            default=["live_intelligence","ticker_signals","market_briefing",
                     "strategy_signals","auto_evaluator"],
        )
    with f3:
        action_filter = st.multiselect(
            "Action", ["BUY","SHORT","HOLD","WATCH"],
            default=["BUY","SHORT","HOLD","WATCH"],
        )
    with f4:
        ticker_search = st.text_input("Search ticker", placeholder="e.g. NVDA")

    df = get_signals_df(
        days_back     = days_back,
        source_filter = source_filter or None,
        action_filter = action_filter or None,
    )

    if ticker_search.strip():
        df = df[df["ticker"].str.upper() == ticker_search.strip().upper()]

    if df.empty:
        st.info(
            "No signals yet. Run Live News Feed, Ticker Signal Lookup, "
            "or Morning Briefing to start building your history."
        )
        return

    stats = get_signal_stats(df)

    st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)
    st.caption(f"Period: {stats.get('date_range','—')}")

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val, color in [
        (m1, "Total",    stats["total"],                          None),
        (m2, "BUY",      stats["by_action"].get("BUY",0),         "#4ade80"),
        (m3, "SHORT",    stats["by_action"].get("SHORT",0),        "#f87171"),
        (m4, "HIGH Conf",stats["by_confidence"].get("HIGH",0),     "#38bdf8"),
        (m5, "HIGH %",   f"{stats['high_conf_pct']}%",            None),
    ]:
        with col:
            c = f'style="color:{color};"' if color else ""
            st.markdown(
                f'<div class="metric-box"><div class="label">{label}</div>'
                f'<div class="value" {c}>{val}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        act = stats["by_action"]
        if act:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
            colors = {"BUY":GREEN,"SHORT":RED,"HOLD":AMBER,"WATCH":BLUE}
            pc = [colors.get(k, MUTED) for k in act.keys()]
            _, _, ats = ax.pie(act.values(), labels=act.keys(), autopct="%1.0f%%",
                               colors=pc, startangle=90,
                               textprops={"color":TEXT,"fontsize":9})
            for at in ats: at.set_color(CARD); at.set_fontsize(8)
            ax.set_title("Signal Breakdown", color=CYAN, fontsize=10)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c2:
        top = stats["top_tickers"]
        if top:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
            ax.barh(list(top.keys())[::-1], list(top.values())[::-1],
                    color=BLUE, edgecolor=BORDER, linewidth=0.5)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_title("Most Signalled Tickers", color=CYAN, fontsize=10)
            ax.set_xlabel("Count", color=MUTED, fontsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.grid(axis="x", color=BORDER, linestyle="--", linewidth=0.5, alpha=0.5)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    if len(df) > 1:
        st.markdown('<div class="section-header">📅 Timeline</div>',
                    unsafe_allow_html=True)
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["timestamp"]).dt.date
        daily = df2.groupby("date").size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(12, 2.5))
        fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
        ax.bar(pd.to_datetime(daily["date"]), daily["count"],
               color=BLUE, width=0.8, edgecolor=BORDER, linewidth=0.3)
        ax.set_title("Signals per Day", color=CYAN, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(axis="y", color=BORDER, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="section-header">📋 All Signals</div>',
                unsafe_allow_html=True)
    dc = ["timestamp","source","ticker","action","confidence",
          "urgency","time_horizon","reasoning","article_title"]
    disp = df[[c for c in dc if c in df.columns]].copy()

    def ca(v): return {"BUY":"color:#4ade80","SHORT":"color:#f87171",
                       "HOLD":"color:#fbbf24","WATCH":"color:#38bdf8"}.get(v,"")
    def cc(v): return {"HIGH":"color:#38bdf8","MEDIUM":"color:#fbbf24",
                       "LOW":"color:#6b8fad"}.get(v,"")
    sty = disp.style
    if "action"     in disp.columns: sty = sty.map(ca, subset=["action"])
    if "confidence" in disp.columns: sty = sty.map(cc, subset=["confidence"])
    st.dataframe(sty, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV", data=csv,
        file_name=f"finsight_signals_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv", use_container_width=True,
    )

    with st.expander("⚠️ Danger Zone"):
        if st.button("🗑️ Clear ALL signal history", key="clear_history"):
            delete_all_signals()
            st.success("Cleared.")
            st.rerun()

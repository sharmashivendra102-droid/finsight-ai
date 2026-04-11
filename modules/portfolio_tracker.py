"""
Portfolio P&L Tracker
======================
- User enters holdings: ticker, shares, average cost
- Fetches live prices via yfinance
- Shows P&L, allocation, and portfolio-level risk flags
- Pulls correlation risk from session state if available
- Persists holdings in SQLite so they survive page refreshes
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).parent.parent / "portfolio.db"

BLUE   = "#38bdf8"
CYAN   = "#7dd3fc"
AMBER  = "#fbbf24"
RED    = "#f87171"
GREEN  = "#4ade80"
CARD   = "#0d1b2a"
BORDER = "#1e3a5f"
TEXT   = "#c9d8e8"
MUTED  = "#6b8fad"
PURPLE = "#a78bfa"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT NOT NULL UNIQUE,
            shares    REAL NOT NULL,
            avg_cost  REAL NOT NULL,
            notes     TEXT,
            added_at  TEXT
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _save_holding(ticker: str, shares: float, avg_cost: float, notes: str = ""):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO holdings (ticker, shares, avg_cost, notes, added_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            shares   = excluded.shares,
            avg_cost = excluded.avg_cost,
            notes    = excluded.notes
    """, (ticker.upper(), shares, avg_cost, notes, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def _delete_holding(ticker: str):
    conn = _get_conn()
    conn.execute("DELETE FROM holdings WHERE ticker = ?", (ticker.upper(),))
    conn.commit()
    conn.close()


def _load_holdings() -> pd.DataFrame:
    try:
        conn = _get_conn()
        df   = pd.read_sql_query("SELECT * FROM holdings ORDER BY ticker", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_prices(tickers: tuple) -> dict:
    """Fetch current and previous close for each ticker."""
    import yfinance as yf
    prices = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="10d", interval="1d")
            if hist.empty:
                continue
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                continue
            prices[ticker] = {
                "current": float(closes.iloc[-1]),
                "prev":    float(closes.iloc[-2]),
                "pct_chg": (float(closes.iloc[-1]) - float(closes.iloc[-2])) / float(closes.iloc[-2]) * 100,
            }
        except Exception:
            continue
    return prices


def _style_fig(fig, ax_list):
    fig.patch.set_facecolor(CARD)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(CYAN)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.5)


def run_portfolio_tracker():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Real-time portfolio P&L — your actual holdings, live prices, personalised risk.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Holdings are saved locally and persist across sessions.
        Prices refresh every 2 minutes. Run Correlation Analytics first to get
        correlation risk specific to your actual portfolio.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Add / Edit holding ─────────────────────────────────────────────────
    with st.expander("➕ Add or update a holding", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 3, 1])
        with c1:
            new_ticker = st.text_input("Ticker", placeholder="e.g. NVDA", key="pt_ticker").strip().upper()
        with c2:
            new_shares = st.number_input("Shares", min_value=0.0001, value=1.0, step=0.01, key="pt_shares", format="%.4f")
        with c3:
            new_cost   = st.number_input("Avg Cost ($)", min_value=0.01, value=100.0, step=0.01, key="pt_cost", format="%.2f")
        with c4:
            new_notes  = st.text_input("Notes (optional)", placeholder="e.g. Long-term hold", key="pt_notes")
        with c5:
            st.markdown("<div style='margin-top:1.8rem;'>", unsafe_allow_html=True)
            add_btn = st.button("💾 Save", key="pt_add", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if add_btn:
            if not new_ticker:
                st.error("Enter a ticker.")
            elif new_shares <= 0:
                st.error("Shares must be > 0.")
            elif new_cost <= 0:
                st.error("Avg cost must be > 0.")
            else:
                _save_holding(new_ticker, new_shares, new_cost, new_notes)
                st.success(f"✅ Saved {new_ticker}: {new_shares} shares @ ${new_cost:.2f}")
                st.rerun()

    # ── Load holdings ──────────────────────────────────────────────────────
    holdings = _load_holdings()

    if holdings.empty:
        st.info("No holdings yet. Add your first position above.")
        return

    tickers = tuple(holdings["ticker"].tolist())

    # ── Fetch prices ───────────────────────────────────────────────────────
    with st.spinner("📡 Fetching live prices…"):
        prices = _fetch_prices(tickers)

    # ── Build P&L table ────────────────────────────────────────────────────
    rows = []
    for _, h in holdings.iterrows():
        ticker   = h["ticker"]
        shares   = float(h["shares"])
        avg_cost = float(h["avg_cost"])
        cost_basis = shares * avg_cost

        p = prices.get(ticker, {})
        current_price = p.get("current", None)
        pct_today     = p.get("pct_chg", None)

        if current_price:
            market_value = shares * current_price
            pnl_dollar   = market_value - cost_basis
            pnl_pct      = (pnl_dollar / cost_basis) * 100
            today_pnl    = shares * current_price * (pct_today / 100) if pct_today is not None else None
        else:
            market_value = None
            pnl_dollar   = None
            pnl_pct      = None
            today_pnl    = None

        rows.append({
            "Ticker":        ticker,
            "Shares":        shares,
            "Avg Cost":      avg_cost,
            "Current Price": current_price,
            "Market Value":  market_value,
            "Cost Basis":    cost_basis,
            "P&L ($)":       pnl_dollar,
            "P&L (%)":       pnl_pct,
            "Today (%)":     pct_today,
            "Today P&L ($)": today_pnl,
            "Notes":         h.get("notes", ""),
        })

    pnl_df = pd.DataFrame(rows)

    # ── Portfolio summary ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">💼 Portfolio Summary</div>', unsafe_allow_html=True)
    st.caption(f"Prices as of {datetime.now().strftime('%H:%M ET')} — updates every 2 minutes")

    valid_rows    = pnl_df[pnl_df["Market Value"].notna()]
    total_value   = valid_rows["Market Value"].sum()
    total_cost    = valid_rows["Cost Basis"].sum()
    total_pnl     = valid_rows["P&L ($)"].sum()
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    today_total   = valid_rows["Today P&L ($)"].sum() if "Today P&L ($)" in valid_rows else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="label">Portfolio Value</div><div class="value">${total_value:,.0f}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="label">Cost Basis</div><div class="value" style="color:{MUTED};">${total_cost:,.0f}</div></div>', unsafe_allow_html=True)
    with m3:
        pnl_color = GREEN if total_pnl >= 0 else RED
        sign = "+" if total_pnl >= 0 else ""
        st.markdown(f'<div class="metric-box"><div class="label">Total P&L</div><div class="value" style="color:{pnl_color};">{sign}${total_pnl:,.0f}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="label">Total P&L %</div><div class="value" style="color:{pnl_color};">{sign}{total_pnl_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m5:
        td_color = GREEN if today_total >= 0 else RED
        td_sign  = "+" if today_total >= 0 else ""
        st.markdown(f'<div class="metric-box"><div class="label">Today P&L</div><div class="value" style="color:{td_color};">{td_sign}${today_total:,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Allocation pie ─────────────────────────────────────────────────────
    if not valid_rows.empty and total_value > 0:
        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig, ax = plt.subplots(figsize=(5, 4))
            _style_fig(fig, ax)
            alloc_vals  = valid_rows["Market Value"].values
            alloc_lbls  = valid_rows["Ticker"].values
            palette = [BLUE, AMBER, GREEN, RED, PURPLE, CYAN, "#fb923c", "#a3e635", "#e879f9", "#67e8f9"]
            colors  = [palette[i % len(palette)] for i in range(len(alloc_vals))]
            wedges, texts, autotexts = ax.pie(
                alloc_vals, labels=alloc_lbls, autopct="%1.1f%%",
                colors=colors, startangle=90,
                textprops={"color": TEXT, "fontsize": 9}
            )
            for at in autotexts:
                at.set_color(CARD)
                at.set_fontsize(8)
            ax.set_title("Portfolio Allocation", color=CYAN, fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_bar:
            # P&L by ticker bar
            fig, ax = plt.subplots(figsize=(5, 4))
            _style_fig(fig, ax)
            pnl_vals   = valid_rows["P&L (%)"].values
            pnl_lbls   = valid_rows["Ticker"].values
            bar_colors = [GREEN if v >= 0 else RED for v in pnl_vals]
            ax.barh(pnl_lbls, pnl_vals, color=bar_colors, edgecolor=BORDER, linewidth=0.5)
            ax.axvline(0, color=MUTED, linewidth=0.8)
            ax.set_title("P&L % by Position", color=CYAN, fontsize=10)
            ax.set_xlabel("Return %", color=MUTED, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Holdings table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Holdings Detail</div>', unsafe_allow_html=True)

    display = pnl_df[["Ticker", "Shares", "Avg Cost", "Current Price",
                       "Market Value", "Cost Basis", "P&L ($)", "P&L (%)", "Today (%)", "Notes"]].copy()

    def fmt_pnl_dollar(v):
        if pd.isna(v): return ""
        return f"color: {GREEN}" if v >= 0 else f"color: {RED}"

    def fmt_pnl_pct(v):
        if pd.isna(v): return ""
        return f"color: {GREEN}" if v >= 0 else f"color: {RED}"

    styler = display.style
    for col in ["P&L ($)", "Today ($)"]:
        if col in display.columns:
            styler = styler.map(fmt_pnl_dollar, subset=[col])
    for col in ["P&L (%)", "Today (%)"]:
        if col in display.columns:
            styler = styler.map(fmt_pnl_pct, subset=[col])

    styler = styler.format({
        "Shares":        "{:.4f}",
        "Avg Cost":      "${:.2f}",
        "Current Price": lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
        "Market Value":  lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A",
        "Cost Basis":    "${:,.0f}",
        "P&L ($)":       lambda x: f"${x:+,.0f}" if pd.notna(x) else "N/A",
        "P&L (%)":       lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
        "Today (%)":     lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
    })
    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Allocation % table ─────────────────────────────────────────────────
    if total_value > 0:
        alloc_table = valid_rows[["Ticker", "Market Value"]].copy()
        alloc_table["Allocation %"] = (alloc_table["Market Value"] / total_value * 100).round(1)
        alloc_table["Market Value"] = alloc_table["Market Value"].apply(lambda x: f"${x:,.0f}")
        alloc_table["Allocation %"] = alloc_table["Allocation %"].apply(lambda x: f"{x:.1f}%")

        # Check concentration risk
        top_alloc = valid_rows.copy()
        top_alloc["alloc_pct"] = top_alloc["Market Value"] / total_value * 100
        top_alloc = top_alloc.sort_values("alloc_pct", ascending=False)

    # ── Personalised risk flags ────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Personalised Risk Flags</div>', unsafe_allow_html=True)

    flags = []

    # Concentration risk
    if total_value > 0:
        top_alloc2 = valid_rows.copy()
        top_alloc2["alloc_pct"] = top_alloc2["Market Value"] / total_value * 100

        over_30 = top_alloc2[top_alloc2["alloc_pct"] > 30]
        if not over_30.empty:
            for _, row in over_30.iterrows():
                flags.append(("risk", f"🔴 Concentration Risk: {row['Ticker']} is {row['alloc_pct']:.1f}% of your portfolio. A single position over 30% represents significant concentration risk."))

        over_20 = top_alloc2[(top_alloc2["alloc_pct"] > 20) & (top_alloc2["alloc_pct"] <= 30)]
        if not over_20.empty:
            for _, row in over_20.iterrows():
                flags.append(("warn", f"🟡 Elevated Weight: {row['Ticker']} is {row['alloc_pct']:.1f}% of your portfolio. Monitor closely."))

    # Correlation risk from session
    corr_df = st.session_state.get("corr_df")
    held_tickers = set(pnl_df["Ticker"].tolist())

    if corr_df is not None and not corr_df.empty:
        relevant_pairs = corr_df[
            corr_df["Ticker 1"].isin(held_tickers) & corr_df["Ticker 2"].isin(held_tickers)
        ]
        high_corr_pairs = relevant_pairs[relevant_pairs["Correlation"] >= 0.75]
        if not high_corr_pairs.empty:
            for _, row in high_corr_pairs.iterrows():
                flags.append(("risk", f"🔴 High Correlation in Portfolio: {row['Ticker 1']} / {row['Ticker 2']} correlation = {row['Correlation']:.2f}. These positions move together — limited diversification between them."))

        neg_pairs = relevant_pairs[relevant_pairs["Correlation"] <= -0.4]
        if not neg_pairs.empty:
            for _, row in neg_pairs.iterrows():
                flags.append(("ok", f"🟢 Natural Hedge: {row['Ticker 1']} / {row['Ticker 2']} correlation = {row['Correlation']:.2f}. These positions partially offset each other."))

    # Today's movers
    big_movers = valid_rows[valid_rows["Today (%)"].abs() > 3] if "Today (%)" in valid_rows.columns else pd.DataFrame()
    if not big_movers.empty:
        for _, row in big_movers.iterrows():
            direction = "up" if row["Today (%)"] > 0 else "down"
            flags.append(("warn", f"⚡ {row['Ticker']} moved {row['Today (%)']:+.1f}% today (${row['Today P&L ($)']:+,.0f} impact on your position)."))

    # Losers flag
    big_losers = valid_rows[valid_rows["P&L (%)"] < -20] if "P&L (%)" in valid_rows.columns else pd.DataFrame()
    if not big_losers.empty:
        for _, row in big_losers.iterrows():
            flags.append(("risk", f"🔴 {row['Ticker']} is down {row['P&L (%)']:.1f}% from your average cost. Consider reviewing your thesis."))

    if not flags:
        st.success("✅ No major risk flags detected in your current portfolio.")
    else:
        for level, msg in flags:
            css = "risk-flag" if level == "risk" else ("ok-flag" if level == "ok" else "insight-card")
            st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)

    # ── Correlation tip ────────────────────────────────────────────────────
    if corr_df is None:
        st.info("💡 Run **Correlation Analytics** with your portfolio tickers to get correlation risk specific to your holdings.")

    # ── Delete holdings ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗑️ Remove Holdings</div>', unsafe_allow_html=True)
    del_ticker = st.selectbox("Select holding to remove", [""] + list(holdings["ticker"]), key="pt_del")
    if st.button("Remove holding", key="pt_del_btn") and del_ticker:
        _delete_holding(del_ticker)
        st.success(f"Removed {del_ticker}")
        st.rerun()

    # ── Export ─────────────────────────────────────────────────────────────
    csv = pnl_df.to_csv(index=False)
    st.download_button("⬇️ Download portfolio as CSV", data=csv,
                       file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv")

"""
Portfolio P&L Tracker — Rich Version
Includes: P&L, allocation, risk flags, technical signals per holding,
correlation matrix for held tickers, news signals from session,
historical performance chart, and sector breakdown.
"""

"""
Portfolio Tracker — Supabase backend
======================================
Holdings are stored in Supabase (Postgres) so they persist
across Streamlit Cloud redeploys permanently.

Supabase setup — run this SQL in your Supabase SQL Editor:

    CREATE TABLE holdings (
      id        BIGSERIAL PRIMARY KEY,
      ticker    TEXT NOT NULL UNIQUE,
      shares    REAL NOT NULL,
      avg_cost  REAL NOT NULL,
      notes     TEXT,
      added_at  TEXT
    );

Then add to Streamlit Secrets:
    SUPABASE_URL = "https://xxxx.supabase.co"
    SUPABASE_KEY = "eyJ..."
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

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

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMD":"Technology",
    "INTC":"Technology","GOOGL":"Technology","GOOG":"Technology","META":"Technology",
    "AMZN":"Consumer Disc.","TSLA":"Consumer Disc.","HD":"Consumer Disc.",
    "JPM":"Financials","GS":"Financials","BAC":"Financials","WFC":"Financials","V":"Financials","MA":"Financials",
    "JNJ":"Healthcare","UNH":"Healthcare","PFE":"Healthcare","MRK":"Healthcare","ABBV":"Healthcare",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy",
    "BA":"Industrials","CAT":"Industrials","GE":"Industrials","UPS":"Industrials","LMT":"Industrials",
    "WMT":"Consumer Staples","PG":"Consumer Staples","KO":"Consumer Staples","PEP":"Consumer Staples",
    "DIS":"Communication","NFLX":"Communication","CMCSA":"Communication","T":"Communication",
    "BTC-USD":"Crypto","ETH-USD":"Crypto","SOL-USD":"Crypto",
    "GLD":"Commodities","SLV":"Commodities","GDX":"Commodities","OIL":"Commodities","CL=F":"Commodities",
    "QQQ":"ETF","SPY":"ETF","VOO":"ETF","VTI":"ETF","IWM":"ETF","VIX":"ETF",
    "TLT":"Bonds","BND":"Bonds","AGG":"Bonds","^GSPC":"Index","^DJI":"Index","^IXIC":"Index",
}

# ── Supabase client ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_client():
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
        key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def _save_holding(ticker, shares, avg_cost, notes=""):
    client = _get_client()
    if client is None:
        st.error("⚠️ Supabase not configured — holdings cannot be saved.")
        return
    try:
        client.table("holdings").upsert({
            "ticker":   ticker.upper(),
            "shares":   shares,
            "avg_cost": avg_cost,
            "notes":    notes or "",
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, on_conflict="ticker").execute()
    except Exception as e:
        st.error(f"Could not save holding: {str(e)[:100]}")


def _delete_holding(ticker):
    client = _get_client()
    if client is None:
        return
    try:
        client.table("holdings").delete().eq("ticker", ticker.upper()).execute()
    except Exception as e:
        st.error(f"Could not delete holding: {str(e)[:100]}")


def _load_holdings():
    client = _get_client()
    if client is None:
        return pd.DataFrame()
    try:
        resp = client.table("holdings").select("*").order("ticker").execute()
        rows = resp.data if resp.data else []
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_holding_count() -> int:
    """Fast count for Command Center status strip."""
    client = _get_client()
    if client is None:
        return 0
    try:
        resp = client.table("holdings").select("id", count="exact").execute()
        return resp.count or 0
    except Exception:
        return 0

@st.cache_data(ttl=120, show_spinner=False)
def _fetch_full_data(tickers: tuple) -> dict:
    """Fetch current price, history, and technicals for each ticker."""
    import yfinance as yf
    result = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="1y", interval="1d")
            if hist.empty:
                continue
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                continue

            current = float(closes.iloc[-1])
            prev    = float(closes.iloc[-2])
            ret     = closes.pct_change().dropna()

            # RSI
            delta = ret
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rsi   = float((100 - (100 / (1 + gain / (loss + 1e-9)))).iloc[-1])

            # Moving averages
            ma50  = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None
            ma200 = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None

            # 52-week high/low
            high52 = float(closes.max())
            low52  = float(closes.min())
            from_high = (current - high52) / high52 * 100
            from_low  = (current - low52)  / low52  * 100

            # Volatility (annualised)
            vol = float(ret.std() * np.sqrt(252) * 100) if len(ret) > 5 else None

            # Beta vs SPY (approximate)
            beta = None
            try:
                spy_hist = yf.Ticker("SPY").history(period="1y", interval="1d")
                spy_ret  = spy_hist["Close"].pct_change().dropna()
                aligned  = pd.concat([ret, spy_ret], axis=1).dropna()
                if len(aligned) > 30:
                    cov  = aligned.cov().iloc[0, 1]
                    var  = aligned.iloc[:, 1].var()
                    beta = round(cov / var, 2) if var > 0 else None
            except Exception:
                pass

            result[ticker] = {
                "current":   current,
                "prev":      prev,
                "pct_today": (current - prev) / prev * 100,
                "closes":    closes,
                "rsi":       rsi,
                "ma50":      ma50,
                "ma200":     ma200,
                "high52":    high52,
                "low52":     low52,
                "from_high": from_high,
                "from_low":  from_low,
                "vol":       vol,
                "beta":      beta,
            }
        except Exception:
            continue
    return result


def _style_fig(fig, ax_list):
    fig.patch.set_facecolor(CARD)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(CYAN)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.4)


def _technical_signal(data: dict) -> tuple:
    """Return (signal, reasoning) based on technicals."""
    rsi    = data.get("rsi")
    ma50   = data.get("ma50")
    ma200  = data.get("ma200")
    current= data.get("current")
    signals = []

    if rsi is not None:
        if rsi < 30:
            signals.append(("BUY", f"RSI {rsi:.0f} — oversold territory"))
        elif rsi > 70:
            signals.append(("SHORT", f"RSI {rsi:.0f} — overbought territory"))
        else:
            signals.append(("NEUTRAL", f"RSI {rsi:.0f} — neutral range"))

    if ma50 and ma200 and current:
        if current > ma50 > ma200:
            signals.append(("BUY", "Price above 50MA & 200MA — bullish trend"))
        elif current < ma50 < ma200:
            signals.append(("SHORT", "Price below 50MA & 200MA — bearish trend"))
        elif ma50 > ma200:
            signals.append(("NEUTRAL", "Golden cross (50MA>200MA) — positive structure"))
        else:
            signals.append(("NEUTRAL", "Death cross (50MA<200MA) — watch closely"))

    if not signals:
        return "NEUTRAL", "Insufficient data"

    # Majority vote
    actions = [s[0] for s in signals]
    if actions.count("BUY") > actions.count("SHORT"):
        overall = "BUY"
    elif actions.count("SHORT") > actions.count("BUY"):
        overall = "SHORT"
    else:
        overall = "NEUTRAL"

    reasoning = " | ".join(s[1] for s in signals)
    return overall, reasoning


def _news_signals_for_ticker(ticker: str) -> list:
    """Pull any signals for this ticker from the session signal history."""
    # Check live intelligence signals in session
    live_signals = []
    processed = st.session_state.get("processed_signals", [])
    ticker_upper = ticker.upper()

    for article, analysis in processed:
        for rec in analysis.get("recommendations", []):
            if rec.get("ticker", "").upper() == ticker_upper:
                live_signals.append({
                    "source":    article.get("source", "Live Feed"),
                    "title":     article.get("title", "")[:80],
                    "action":    rec.get("action", "WATCH"),
                    "confidence":rec.get("confidence", "LOW"),
                    "reasoning": rec.get("reasoning", ""),
                    "horizon":   rec.get("time_horizon", ""),
                })

    # Also check ticker analysis results
    ticker_results = st.session_state.get("ticker_analysis_results", [])
    for r in ticker_results:
        if r.get("ticker", "").upper() == ticker_upper:
            for sig in r.get("signals", []):
                live_signals.append({
                    "source":    "Ticker Analysis",
                    "title":     sig.get("source_article", "")[:80],
                    "action":    sig.get("action", "WATCH"),
                    "confidence":sig.get("confidence", "LOW"),
                    "reasoning": sig.get("reasoning", ""),
                    "horizon":   sig.get("time_horizon", ""),
                })

    return live_signals[:5]   # max 5 most recent


def run_portfolio_tracker():

    st.markdown("""
    <div class="insight-card">
        <b style="color:#7dd3fc;">Real-time portfolio tracker — live prices, technicals, news signals, and personalised risk.</b><br>
        <span style="color:#6b8fad;font-size:0.9rem;">
        Holdings persist across sessions. Prices and technicals refresh every 2 minutes.
        Run Live Intelligence and Ticker Signals to populate news signals per holding.
        Run Correlation Analytics with your tickers for correlation risk.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Add / Edit ─────────────────────────────────────────────────────────
    with st.expander("➕ Add or update a holding", expanded=False):
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 3, 1])
        with c1: new_ticker = st.text_input("Ticker", placeholder="NVDA", key="pt_ticker").strip().upper()
        with c2: new_shares = st.number_input("Shares", min_value=0.0001, value=1.0, step=0.01, key="pt_shares", format="%.4f")
        with c3: new_cost   = st.number_input("Avg Cost ($)", min_value=0.01, value=100.0, step=0.01, key="pt_cost", format="%.2f")
        with c4: new_notes  = st.text_input("Notes", placeholder="Long-term hold", key="pt_notes")
        with c5:
            st.markdown("<div style='margin-top:1.8rem;'>", unsafe_allow_html=True)
            add_btn = st.button("💾 Save", key="pt_add", use_container_width=True)

        if add_btn:
            if not new_ticker:        st.error("Enter a ticker.")
            elif new_shares <= 0:     st.error("Shares must be > 0.")
            elif new_cost <= 0:       st.error("Avg cost must be > 0.")
            else:
                _save_holding(new_ticker, new_shares, new_cost, new_notes)
                st.success(f"✅ Saved {new_ticker}")
                st.rerun()

    holdings = _load_holdings()
    if holdings.empty:
        st.info("No holdings yet. Add your first position above.")
        return

    tickers = tuple(holdings["ticker"].tolist())

    with st.spinner("📡 Fetching live prices and technicals…"):
        data = _fetch_full_data(tickers)

    st.caption(f"Data as of {datetime.now().strftime('%H:%M ET')} · Refreshes every 2 min · {len(data)}/{len(tickers)} tickers loaded")

    # ── Build master table ─────────────────────────────────────────────────
    rows = []
    for _, h in holdings.iterrows():
        ticker   = h["ticker"]
        shares   = float(h["shares"])
        avg_cost = float(h["avg_cost"])
        cost_basis = shares * avg_cost
        d = data.get(ticker, {})
        current = d.get("current")

        if current:
            mktval   = shares * current
            pnl_d    = mktval - cost_basis
            pnl_pct  = pnl_d / cost_basis * 100
            today_pct= d.get("pct_today", 0)
            today_pnl= shares * current * (today_pct / 100)
        else:
            mktval = pnl_d = pnl_pct = today_pct = today_pnl = None

        tech_signal, tech_reason = _technical_signal(d) if d else ("N/A", "No data")
        sector = SECTOR_MAP.get(ticker, "Other")

        rows.append({
            "Ticker":       ticker,
            "Sector":       sector,
            "Shares":       shares,
            "Avg Cost":     avg_cost,
            "Price":        current,
            "Mkt Value":    mktval,
            "Cost Basis":   cost_basis,
            "P&L ($)":      pnl_d,
            "P&L (%)":      pnl_pct,
            "Today (%)":    today_pct,
            "Today ($)":    today_pnl,
            "RSI":          d.get("rsi"),
            "MA50":         d.get("ma50"),
            "MA200":        d.get("ma200"),
            "52W High":     d.get("high52"),
            "52W Low":      d.get("low52"),
            "From 52W High":d.get("from_high"),
            "Volatility":   d.get("vol"),
            "Beta":         d.get("beta"),
            "Tech Signal":  tech_signal,
            "Tech Reason":  tech_reason,
            "Notes":        h.get("notes", ""),
        })

    df = pd.DataFrame(rows)
    valid = df[df["Mkt Value"].notna()]

    total_val  = valid["Mkt Value"].sum()
    total_cost = valid["Cost Basis"].sum()
    total_pnl  = valid["P&L ($)"].sum()
    total_pct  = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    today_tot  = valid["Today ($)"].sum() if "Today ($)" in valid else 0

    # ── Portfolio Summary ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">💼 Portfolio Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    pnl_color = GREEN if total_pnl >= 0 else RED
    td_color  = GREEN if today_tot >= 0 else RED
    sign = "+" if total_pnl >= 0 else ""
    tds  = "+" if today_tot >= 0 else ""

    with m1: st.markdown(f'<div class="metric-box"><div class="label">Portfolio Value</div><div class="value">${total_val:,.0f}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-box"><div class="label">Cost Basis</div><div class="value" style="color:{MUTED};">${total_cost:,.0f}</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-box"><div class="label">Total P&L ($)</div><div class="value" style="color:{pnl_color};">{sign}${total_pnl:,.0f}</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-box"><div class="label">Total P&L %</div><div class="value" style="color:{pnl_color};">{sign}{total_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m5: st.markdown(f'<div class="metric-box"><div class="label">Today P&L</div><div class="value" style="color:{td_color};">{tds}${today_tot:,.0f}</div></div>', unsafe_allow_html=True)
    with m6: st.markdown(f'<div class="metric-box"><div class="label">Positions</div><div class="value">{len(df)}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Charts row 1: allocation + P&L ────────────────────────────────────
    if not valid.empty and total_val > 0:
        ch1, ch2 = st.columns(2)
        palette = [BLUE, AMBER, GREEN, RED, PURPLE, CYAN, "#fb923c", "#a3e635", "#e879f9", "#67e8f9"]

        with ch1:
            fig, ax = plt.subplots(figsize=(5, 4))
            _style_fig(fig, ax)
            vals   = valid["Mkt Value"].values
            lbls   = valid["Ticker"].values
            colors = [palette[i % len(palette)] for i in range(len(vals))]
            _, texts, autotexts = ax.pie(vals, labels=lbls, autopct="%1.1f%%",
                colors=colors, startangle=90, textprops={"color": TEXT, "fontsize": 8})
            for at in autotexts: at.set_color(CARD); at.set_fontsize(7)
            ax.set_title("Allocation by Position", color=CYAN, fontsize=10)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        with ch2:
            fig, ax = plt.subplots(figsize=(5, 4))
            _style_fig(fig, ax)
            pnl_v = valid["P&L (%)"].values
            pnl_l = valid["Ticker"].values
            bc    = [GREEN if v >= 0 else RED for v in pnl_v]
            ax.barh(pnl_l, pnl_v, color=bc, edgecolor=BORDER, linewidth=0.4)
            ax.axvline(0, color=MUTED, linewidth=0.8)
            ax.set_title("P&L % by Position", color=CYAN, fontsize=10)
            ax.set_xlabel("Return %", color=MUTED, fontsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Sector breakdown ───────────────────────────────────────────────────
    if not valid.empty and total_val > 0:
        st.markdown('<div class="section-header">🏭 Sector Breakdown</div>', unsafe_allow_html=True)
        sector_df = valid.groupby("Sector")["Mkt Value"].sum().reset_index()
        sector_df["Allocation %"] = (sector_df["Mkt Value"] / total_val * 100).round(1)
        sector_df = sector_df.sort_values("Allocation %", ascending=False)

        ch3, ch4 = st.columns([2, 3])
        with ch3:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            _style_fig(fig, ax)
            sc_colors = [palette[i % len(palette)] for i in range(len(sector_df))]
            ax.barh(sector_df["Sector"].values[::-1],
                    sector_df["Allocation %"].values[::-1],
                    color=sc_colors[::-1], edgecolor=BORDER, linewidth=0.4)
            ax.set_xlabel("Allocation %", color=MUTED, fontsize=8)
            ax.set_title("By Sector", color=CYAN, fontsize=10)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        with ch4:
            st.dataframe(
                sector_df.style.format({"Mkt Value": "${:,.0f}", "Allocation %": "{:.1f}%"}),
                use_container_width=True, hide_index=True
            )

    # ── Historical performance chart ───────────────────────────────────────
    st.markdown('<div class="section-header">📈 Historical Performance (vs SPY)</div>', unsafe_allow_html=True)
    st.caption("Shows how your portfolio would have performed over 1 year at current weights vs SPY benchmark.")

    if not valid.empty and total_val > 0:
        try:
            import yfinance as yf
            # Build portfolio return series
            portfolio_ret = None
            for _, row in valid.iterrows():
                ticker = row["Ticker"]
                weight = row["Mkt Value"] / total_val
                d      = data.get(ticker, {})
                closes = d.get("closes")
                if closes is None or len(closes) < 2:
                    continue
                ret = closes.pct_change().dropna()
                if portfolio_ret is None:
                    portfolio_ret = ret * weight
                else:
                    portfolio_ret, ret = portfolio_ret.align(ret, join="inner")
                    portfolio_ret = portfolio_ret + ret * weight

            if portfolio_ret is not None and len(portfolio_ret) > 5:
                cum_port = (1 + portfolio_ret).cumprod()

                # SPY benchmark
                spy_hist = yf.Ticker("SPY").history(period="1y", interval="1d")
                spy_ret  = spy_hist["Close"].pct_change().dropna()
                cum_spy, cum_port = (1 + spy_ret).cumprod().align((1 + portfolio_ret).cumprod(), join="inner")

                fig, ax = plt.subplots(figsize=(12, 4))
                _style_fig(fig, ax)
                ax.plot(cum_port.index, (cum_port - 1) * 100, color=BLUE, linewidth=2, label="Your Portfolio")
                ax.plot(cum_spy.index,  (cum_spy - 1)  * 100, color=AMBER, linewidth=1.5, linestyle="--", label="SPY (Benchmark)", alpha=0.8)
                ax.axhline(0, color=MUTED, linewidth=0.7, linestyle=":")
                ax.fill_between(cum_port.index, (cum_port-1)*100, 0,
                    where=cum_port>=1, alpha=0.06, color=GREEN)
                ax.fill_between(cum_port.index, (cum_port-1)*100, 0,
                    where=cum_port<1,  alpha=0.06, color=RED)
                ax.set_title("1-Year Portfolio Performance vs SPY", color=CYAN)
                ax.set_ylabel("Return %", color=MUTED)
                ax.legend(fontsize=9, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.xticks(rotation=30)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as e:
            st.caption(f"Performance chart unavailable: {str(e)[:60]}")

    # ── Per-position deep dive ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Per-Position Analysis</div>', unsafe_allow_html=True)

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        d      = data.get(ticker, {})

        _pnl_pct = row['P&L (%)'] if pd.notna(row['P&L (%)']) else None
        _label = f"**{ticker}** — {row['Sector']}  ·  P&L: {'+' if (_pnl_pct or 0) >= 0 else ''}{_pnl_pct:.1f}%  ·  Tech: {row['Tech Signal']}" if _pnl_pct is not None else f"**{ticker}** — {row['Sector']}"
        with st.expander(_label, expanded=False):

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**💰 Position**")
                price_str = f"${row['Price']:.2f}" if row['Price'] else "N/A"
                mktval_str = f"${row['Mkt Value']:,.0f}" if row['Mkt Value'] else "N/A"
                pnl_str = f"{'+' if (row['P&L ($)'] or 0)>=0 else ''}${row['P&L ($)']:,.0f} ({row['P&L (%)']:+.1f}%)" if row['P&L ($)'] is not None else "N/A"
                today_str = f"{row['Today (%)']:+.1f}% (${row['Today ($)']:+,.0f})" if row['Today (%)'] is not None else "N/A"
                st.markdown(f"""
- **Price:** {price_str}
- **Shares:** {row['Shares']:.4f}
- **Avg Cost:** ${row['Avg Cost']:.2f}
- **Market Value:** {mktval_str}
- **Total P&L:** {pnl_str}
- **Today:** {today_str}
                """)

            with col_b:
                st.markdown("**📊 Technicals**")
                rsi = row["RSI"] if pd.notna(row["RSI"]) else None
                ma50 = row.get("MA50")
                ma200 = row.get("MA200")
                high52 = row.get("52W High")
                low52 = row.get("52W Low")
                vol = row.get("Volatility")
                beta = row.get("Beta")

                rsi_str = f"{rsi:.0f}" if rsi else "N/A"
                rsi_label = " (oversold 🟢)" if rsi and rsi < 30 else (" (overbought 🔴)" if rsi and rsi > 70 else "")
                ma_signal = ""
                if ma50 and ma200:
                    ma_signal = " (Golden Cross 🟢)" if ma50 > ma200 else " (Death Cross 🔴)"

                st.markdown(f"""
- **RSI (14):** {rsi_str}{rsi_label}
- **50-Day MA:** ${ma50:.2f}{ma_signal if ma50 else ""} {" ✅ above" if row['Price'] and ma50 and row['Price'] > ma50 else " ❌ below" if row['Price'] and ma50 else ""}
- **200-Day MA:** ${ma200:.2f} {" ✅ above" if row['Price'] and ma200 and row['Price'] > ma200 else " ❌ below" if row['Price'] and ma200 else ""}
- **52W High:** {"$"+f"{high52:.2f}" if high52 else "N/A"} ({f"{row.get('From 52W High', None):+.1f}%" if row.get('From 52W High') is not None else "N/A"})
- **52W Low:** {"$"+f"{low52:.2f}" if low52 else "N/A"} ({f"{row.get('From 52W Low', None):+.1f}%" if row.get('From 52W Low') is not None else "N/A"})
- **Volatility:** {f"{vol:.1f}%" if vol else "N/A"} annualised
- **Beta:** {beta if beta else "N/A"}
                """ if ma50 and ma200 else "Data loading…")

                # Technical signal badge
                sig_color = {"BUY": GREEN, "SHORT": RED, "NEUTRAL": AMBER}.get(row["Tech Signal"], MUTED)
                st.markdown(f"""
                <div style="background:{sig_color}22;border:1px solid {sig_color}44;border-radius:8px;
                            padding:0.5rem 0.8rem;margin-top:0.5rem;">
                    <b style="color:{sig_color};">Technical: {row['Tech Signal']}</b><br>
                    <span style="color:#c9d8e8;font-size:0.8rem;">{row['Tech Reason']}</span>
                </div>
                """, unsafe_allow_html=True)

            with col_c:
                st.markdown("**📰 News Signals**")
                news_sigs = _news_signals_for_ticker(ticker)
                if news_sigs:
                    for ns in news_sigs:
                        action = ns["action"]
                        dot    = {"BUY":"🟢","SHORT":"🔴","HOLD":"🟡","WATCH":"🔵"}.get(action,"⚪")
                        conf   = {"HIGH":"🔵","MEDIUM":"🟡","LOW":"🔘"}.get(ns["confidence"],"🔘")
                        st.markdown(f"""
                        <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;
                                    padding:0.5rem 0.8rem;margin-bottom:0.4rem;font-size:0.8rem;">
                            <b style="color:#c9d8e8;">{dot} {action}</b> {conf} {ns['confidence']}<br>
                            <span style="color:#94a3b8;">{ns['reasoning'][:100]}</span><br>
                            <span style="color:#6b8fad;font-size:0.72rem;">📰 {ns['title'][:60]} · ⏱ {ns['horizon']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No news signals yet. Run Live Intelligence or Ticker Signals.")

                if row.get("Notes"):
                    st.caption(f"📝 {row['Notes']}")

            # Mini price chart
            closes = d.get("closes")
            if closes is not None and len(closes) > 20:
                fig, ax = plt.subplots(figsize=(10, 2))
                _style_fig(fig, ax)
                ax.plot(closes.index, closes.values, color=BLUE, linewidth=1.2)
                # Add MA lines if available
                if ma50:
                    ma50_series = closes.rolling(50).mean()
                    ax.plot(ma50_series.index, ma50_series.values, color=AMBER, linewidth=0.8, linestyle="--", label="50MA", alpha=0.8)
                if ma200:
                    ma200_series = closes.rolling(200).mean()
                    ax.plot(ma200_series.index, ma200_series.values, color=RED, linewidth=0.8, linestyle="--", label="200MA", alpha=0.8)
                ax.set_title(f"{ticker} — 1 Year Price", color=CYAN, fontsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
                plt.xticks(rotation=20, fontsize=7)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Holdings table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Holdings Table</div>', unsafe_allow_html=True)

    table_cols = ["Ticker","Sector","Price","Mkt Value","P&L ($)","P&L (%)","Today (%)","RSI","Beta","Volatility","Tech Signal"]
    display = df[[c for c in table_cols if c in df.columns]].copy()

    def c_pnl(v):
        if pd.isna(v): return ""
        return f"color:{GREEN}" if v >= 0 else f"color:{RED}"
    def c_sig(v):
        return {"BUY": f"color:{GREEN}", "SHORT": f"color:{RED}", "NEUTRAL": f"color:{AMBER}"}.get(v, "")

    styler = display.style
    for col in ["P&L ($)", "P&L (%)", "Today (%)"]:
        if col in display.columns: styler = styler.map(c_pnl, subset=[col])
    if "Tech Signal" in display.columns: styler = styler.map(c_sig, subset=["Tech Signal"])
    styler = styler.format({
        "Price":      lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
        "Mkt Value":  lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A",
        "P&L ($)":    lambda x: f"${x:+,.0f}" if pd.notna(x) else "N/A",
        "P&L (%)":    lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
        "Today (%)":  lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
        "RSI":        lambda x: f"{x:.0f}" if pd.notna(x) else "N/A",
        "Volatility": lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A",
        "Beta":       lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
    })
    st.dataframe(styler, use_container_width=True, hide_index=True)

    # ── Risk flags ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Personalised Risk Flags</div>', unsafe_allow_html=True)
    flags = []

    if total_val > 0:
        for _, row in valid.iterrows():
            alloc = row["Mkt Value"] / total_val * 100
            if alloc > 30:
                flags.append(("risk", f"🔴 **{row['Ticker']}** is {alloc:.1f}% of your portfolio — high concentration risk."))
            elif alloc > 20:
                flags.append(("warn", f"🟡 **{row['Ticker']}** is {alloc:.1f}% of your portfolio — elevated weight."))

    corr_df = st.session_state.get("corr_df")
    held = set(df["Ticker"].tolist())
    if corr_df is not None and not corr_df.empty:
        rel = corr_df[corr_df["Ticker 1"].isin(held) & corr_df["Ticker 2"].isin(held)]
        for _, r in rel[rel["Correlation"] >= 0.75].iterrows():
            flags.append(("risk", f"🔴 **{r['Ticker 1']}/{r['Ticker 2']}** correlation = {r['Correlation']:.2f} — limited diversification between these positions."))
        for _, r in rel[rel["Correlation"] <= -0.4].iterrows():
            flags.append(("ok", f"🟢 **{r['Ticker 1']}/{r['Ticker 2']}** correlation = {r['Correlation']:.2f} — natural hedge."))

    for _, row in df.iterrows():
        if row["Today (%)"] if pd.notna(row["Today (%)"]) else None and abs(row["Today (%)"]) > 3:
            flags.append(("warn", f"⚡ **{row['Ticker']}** moved {row['Today (%)']:+.1f}% today (${row['Today ($)']:+,.0f} impact)."))
        if row["P&L (%)"] if pd.notna(row["P&L (%)"]) else None and row["P&L (%)"] < -20:
            flags.append(("risk", f"🔴 **{row['Ticker']}** is down {row['P&L (%)']:.1f}% from avg cost — review your thesis."))
        rsi = row["RSI"] if pd.notna(row["RSI"]) else None
        if rsi and rsi > 75:
            flags.append(("warn", f"🟡 **{row['Ticker']}** RSI = {rsi:.0f} — overbought. Consider trimming or hedging."))
        if rsi and rsi < 25:
            flags.append(("ok", f"🟢 **{row['Ticker']}** RSI = {rsi:.0f} — deeply oversold. Potential entry or add opportunity."))

    if not flags:
        st.success("✅ No major risk flags detected.")
    else:
        for level, msg in flags:
            css = "risk-flag" if level == "risk" else ("ok-flag" if level == "ok" else "insight-card")
            st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)

    if corr_df is None:
        st.info("💡 Run **Correlation Analytics** with your tickers for correlation-specific risk flags.")

    # ── Remove & Export ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗑️ Manage Holdings</div>', unsafe_allow_html=True)
    del_ticker = st.selectbox("Remove a holding", [""] + list(holdings["ticker"]), key="pt_del")
    if st.button("Remove", key="pt_del_btn") and del_ticker:
        _delete_holding(del_ticker)
        st.success(f"Removed {del_ticker}")
        st.rerun()

    csv = df.to_csv(index=False)
    st.download_button("⬇️ Export portfolio CSV", data=csv,
        file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

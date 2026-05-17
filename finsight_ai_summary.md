# FinSight AI — Full App Summary

> **Version:** 1.0 · **Stack:** Python / Streamlit / Groq (LLaMA 3.3 70B) / Supabase / yfinance  
> **Disclaimer:** Not financial advice. For research and educational use.

---

## What Is It?

FinSight AI is a self-contained financial intelligence platform built in Streamlit. It combines real-time news scanning, AI-generated trade signals, strategy backtesting, live signal generation, portfolio analytics, and a persistent accuracy-tracking loop — all wired together so that every signal the system ever generates is logged to a Supabase database and later evaluated against real market prices.

The core loop is: **generate signal → log to Supabase → evaluate outcome → feed back into profitability engine → rank which strategies to keep using.** Over time the system becomes self-aware of which of its own outputs are actually useful.

---

## Architecture Overview

```
app.py  (17-tab Streamlit shell)
│
├── modules/
│   ├── live_news_engine.py          RSS polling + Groq signal generation
│   ├── ticker_analysis_engine.py    On-demand ticker deep-dive
│   ├── market_summary_engine.py     Pre-market briefing (prices + news)
│   ├── backtest_engine.py           4 strategies, OOS only, walk-forward
│   ├── strategy_signals.py          Live signals + accuracy tracker (NEW)
│   ├── auto_evaluator.py            Walk-forward historical simulation
│   ├── sentiment_engine.py          Article URL → 5-dimension scoring
│   ├── correlation_engine.py        Pairwise + rolling + RF predictor
│   ├── portfolio_tracker.py         Live P&L, technicals, news overlay
│   ├── portfolio_risk.py            Composite risk score
│   ├── advisor_engine.py            Conversational AI advisor
│   ├── signal_history.py            Supabase read/write layer
│   ├── signal_performance.py        Full track record evaluator
│   ├── signal_profitability.py      Strategy auto-ranker
│   ├── news_signal_evaluator.py     News-specific accuracy evaluator
│   ├── eval_core.py                 Shared evaluation engine
│   └── email_alerts.py             Gmail / SendGrid HIGH-urgency alerts
│
├── supabase_test.py                 Diagnostic tool
└── requirements.txt
```

**External services:**
- **Groq API** — LLaMA 3.3 70B for all AI analysis (signals, sentiment, briefings, advisor)
- **Supabase** — Postgres backend for signal history and portfolio holdings
- **Yahoo Finance** (`yfinance`) — all price data, OHLC, technicals
- **RSS feeds** — Reuters, CNBC, Yahoo Finance, MarketWatch, WSJ, Investing.com, Seeking Alpha, FT (10 sources)

---

## The 17 Tabs

### 0 — ⌂ Command Center
The home screen. Shows four live status metrics (feed running, total signals logged, portfolio positions, engine online/offline), then lays out the full daily workflow as clickable tiles. Each tile navigates directly to the relevant tab via a JS injection into the Streamlit tab bar. Also contains the Quick Start Guide (6-step onboarding).

**Key data shown:** live feed status, Supabase signal count, portfolio position count, Groq engine status.

---

### 1 — ☕ Morning Briefing (`market_summary_engine.py`)

Runs every morning before the market opens. Fetches:
- Live prices for 8 major indices/instruments (S&P 500, Nasdaq, Dow, VIX, 10Y yield, Gold, WTI Oil, BTC)
- All 11 sector ETF prices (XLK, XLE, XLF, XLV, XLI, XLY, XLP, XLB, XLRE, XLU, XLC)
- Up to 25 overnight articles from 8 RSS sources

Sends everything to Groq which returns a structured JSON briefing containing:

| Field | Description |
|---|---|
| `market_bias` | BULLISH / BEARISH / NEUTRAL / MIXED |
| `bias_confidence` | HIGH / MEDIUM / LOW |
| `one_line_summary` | Single punchy sentence |
| `key_themes` | 2–4 themes with impact and affected sectors |
| `sector_outlook` | 4–6 sectors with bias and reasoning |
| `tickers_to_watch` | 3–5 tickers with action, confidence, catalyst, risk |
| `watchlist_signals` | User's personal watchlist tickers with signals |
| `levels_to_watch` | Key price levels from actual market data (never invented) |
| `macro_risks` | Top 3 macro risks today |
| `morning_playbook` | 3–4 sentence actionable summary |

All `tickers_to_watch` and `watchlist_signals` are automatically logged to Supabase. Cached for 5 minutes. Prices use the two most recent non-NaN closes to handle pre/post market data gaps.

---

### 2 — 📡 Live News Feed (`live_news_engine.py`)

The core real-time signal engine. Polls 10 RSS sources on a configurable interval (15–120 seconds, default 30s). Age filter (1–72 hours) prevents stale articles from being analysed.

**Flow per refresh:**
1. Parse all feeds, extract title + summary + URL + publication timestamp
2. Deduplicate by URL hash
3. Filter to articles not yet seen in this session
4. Batch up to 5 new articles → send to Groq
5. Groq returns a JSON array — one analysis object per article
6. Each analysis is rendered as a signal card and logged to Supabase

**Per-article analysis includes:**
- `market_impact`: BULLISH / BEARISH / NEUTRAL
- `urgency`: HIGH / MEDIUM / LOW
- `recommendations`: array of `{ticker, action, confidence, time_horizon, reasoning}`
- `affected_sectors`: list of sectors
- `key_risk`: biggest risk from this article

**HIGH urgency** articles trigger email alerts automatically (Gmail or SendGrid). All signals are logged to Supabase in the background. The processed article pool is shared with the Ticker Signal Lookup tab so that tab gets richer signals when the feed has been running.

**Controls:** Start / Stop / Clear buttons, impact and urgency filter dropdowns.

---

### 3 — 🎯 Ticker Signal Lookup (`ticker_analysis_engine.py`)

On-demand deep-dive for any ticker(s). User types one or more tickers (or clicks quick-example chips for TSLA, NVDA, BTC-USD, AAPL, XLE, SPY, META).

**Two-stage article harvesting:**
1. **Live Feed pool first** — articles already processed by Tab 2 this session (highest quality, already AI-analysed)
2. **Fresh RSS fetch** — pulls from all 10 feeds in real time

**Alias matching** — a dictionary of 40+ tickers maps each to company names, exec names, product names, and related terms. NVDA matches "nvidia", "jensen huang", "h100", "blackwell", "cuda", etc. This catches indirect mentions that a simple ticker search would miss.

**Three-pass filtering:**
- Pass 1: direct ticker/alias matches (labelled DIRECT)
- Pass 2: macro context articles (up to 6, labelled MACRO)
- Live Feed articles carry their prior AI analysis as additional context

**Per-ticker output:**
- Overall signal: BUY / SHORT / HOLD / WATCH
- Overall confidence: HIGH / MEDIUM / LOW (with strict rules — LOW is given when only macro context exists, not inflated)
- Market impact, urgency
- Individual signals array with cited article titles
- Bull case / Bear case / Key risk
- Expandable list of which articles were used

All signals logged to Supabase. Results cached in session state so previous analysis persists until refresh.

---

### 4 — 💼 Portfolio Tracker (`portfolio_tracker.py`)

Persistent portfolio manager backed by Supabase `holdings` table (survives redeploys).

**Holdings operations:** Add/update (UPSERT by ticker), delete, export CSV.

**Per-position data (refreshed every 2 minutes):**
- Live price, daily % change, daily P&L
- Total P&L in $ and %
- RSI (14-period)
- 50-day and 200-day moving averages with Golden/Death Cross detection
- 52-week high and low, % distance from each
- Annualised volatility
- Beta vs SPY

**Aggregate portfolio view:**
- Total market value, cost basis, unrealised P&L
- Allocation pie chart
- P&L % bar chart per position
- Sector breakdown (11 sectors, bar + table)
- 1-year portfolio return vs SPY benchmark chart (weighted by current allocation)

**News signal overlay:** For each holding, pulls matching signals from (a) the Live News Feed session and (b) any recent Ticker Signal Lookup results. Shows action, confidence, reasoning, time horizon per signal.

**Personalised risk flags:** Concentration warnings (>30% in one position), correlation risk (if Correlation Analytics has been run with those tickers), RSI overbought/oversold alerts, large daily move alerts, >20% drawdown from cost basis.

---

### 5 — 🔬 Strategy Backtester (`backtest_engine.py`)

Four academically-grounded strategies, all out-of-sample only, with realistic transaction costs.

**Strategies:**

| Strategy | Logic | Best For |
|---|---|---|
| MA Crossover | Buy when fast MA > slow MA, short otherwise. 2 parameters. Zero ML. | Index ETFs (SPY, QQQ, GLD) |
| RSI Mean Reversion | Buy RSI < oversold, short RSI > overbought, exit near 50. Rule-based. | Volatile assets (TSLA, BTC-USD, NVDA) |
| Momentum | Long when N-day return > +2%, short when < -2%. Rebalances every H days. | Multi-asset trend following |
| ML Weekly RF | Random Forest predicting weekly direction. Walk-forward TimeSeriesSplit. | Any ticker with 3+ years history |

**Anti-overfitting measures:**
- Walk-forward `TimeSeriesSplit` — no shuffling, no future data leakage
- All metrics computed on OOS data only
- Realistic transaction costs (default 0.05% per trade)
- Buy-and-hold benchmark comparison shown alongside
- Bootstrap Sharpe confidence intervals (200 samples)

**Output metrics:** Total return (strategy vs B&H), Sharpe ratio with 90% CI, max drawdown, win rate, profit factor, trade count, OOS date range. Three charts: cumulative return, drawdown, monthly returns bar chart. Honest warnings when Sharpe is negative or B&H beats the strategy.

---

### 6 — 🔮 Live Trading Signals (`strategy_signals.py`) ← UPDATED

Applies the same strategy logic as the backtester to current market data to generate a forward-looking trade signal.

**Signal output per ticker:**
- Direction: BUY / SHORT / HOLD
- Entry price (current), stop loss, take profit
- Risk/reward ratio
- Exact indicator values driving the signal (MA values, RSI, momentum %, model probability)
- "Signal flip" price — what price would reverse the signal
- Chart with MA overlays / RSI panel / momentum bars
- Trade plan summary card

Signals are automatically logged to Supabase (`source = "strategy_signals"`) with strategy name, reasoning, and time horizon.

**NEW — Signal Accuracy Tracker (bottom of this tab):**

Loads every strategy signal from Supabase and evaluates it against real market prices using `eval_core`. Shows:

- **Directional accuracy** — did price move ≥ N% in the signal's direction at any point during the hold, using intraday High/Low? (MFE-based, threshold adjustable via slider)
- **Exit accuracy** — was price up at the fixed-horizon exit timestamp?
- **Avg return** with t-test p-value and statistical significance flag
- **Std dev and variance** of returns
- **Sharpe ratio** (rough annualised)
- **Win rate, profit factor, best/worst signal**
- **Return distribution chart** with normal fit overlay (green bars = profitable, red bars = losses)
- **Cumulative P&L curve** (chronological)
- **Rolling accuracy chart** (10-signal window)
- **Per-strategy table** with exit acc %, directional acc %, avg return %, std dev, variance, Sharpe, p-value, significance flag
- **Per-ticker table** with same metrics plus BUY/SHORT count split
- **Variance & risk profile** — skewness, kurtosis, coefficient of variation with explanations
- **Box plot** of return spread: BUY vs SHORT
- **MFE/MAE scatter plot** — visualises which signals were right in direction but wrong in timing
- **Full evaluated signals table** with download CSV

Black-swan events (stock moved 3× normal daily range during eval window) are automatically detected and excluded from accuracy stats, shown separately.

---

### 7 — 🧪 Strategy Simulator (`auto_evaluator.py`)

Walk-forward historical simulation across multiple tickers and strategies simultaneously.

Unlike the backtester (which shows performance of a single strategy), this runs all 4 strategies on all input tickers over a configurable evaluation window and measures accuracy at 5 horizons: 1, 3, 5, 10, and 21 trading days.

**Output:**
- Accuracy heatmap: strategy rows × horizon columns (colour-coded green/amber/red)
- Statistical significance table (t-test, p-values) — tells you if the edge is real or luck
- "Best Performing Combinations" ranked table (statistically significant only)
- Equity curves for each strategy × ticker combo vs SPY benchmark
- Daily signal activity bar chart

All simulated signals are saved to Supabase (`source = "auto_evaluator"`) so the Profitability Engine can include them in its cross-source analysis.

---

### 8 — 📰 News Sentiment (`sentiment_engine.py`)

User pastes one or more article URLs. For each URL:
1. Scrapes full article text (newspaper3k primary, BeautifulSoup fallback)
2. Sends first 3,500 chars to Groq
3. Groq returns 5 scores (0.0–1.0)

| Score | Meaning | High = Bad? |
|---|---|---|
| `political_bias` | How politically slanted is the coverage | ✓ |
| `truthfulness` | How factual and verified | ✗ (high is good) |
| `propaganda` | Overt narrative pushing | ✓ |
| `hype` | Exaggeration / sensationalism | ✓ |
| `panic` | Fear-inducing tone | ✓ |

Results saved to Supabase and session state. Fed into Portfolio Risk and AI Portfolio Advisor as context. Bar chart of average scores. Colour-coded table (green/amber/red per score).

---

### 9 — 🔍 News Signal Accuracy (`news_signal_evaluator.py`)

Evaluates only news-based signals (Live Feed, Ticker Lookup, Morning Briefing) using `eval_core`. Filterable by confidence, source, and date range.

Shows directional vs exit accuracy comparison, rolling accuracy over time, per-source and per-confidence breakdown, statistical significance, and a full evaluated signals table. Flags signals where the thesis was right but timing was wrong (MFE ≥ threshold but exit was negative).

---

### 10 — 📈 Signal Performance (`signal_performance.py`)

Complete signal track record across ALL sources. Uses `eval_core` with the full signal timeline for supersession detection.

Key design choices in `eval_core.py`:
- **Entry = OPEN of next trading day** after signal (not close, not signal time) — the first price a real trader could actually get
- **INTRADAY horizon** = entry OPEN → same-day CLOSE
- **SWING** = 5 trading days, **MEDIUM** = 14 td, **LONG** = 30 td
- **Supersession** — if an opposing signal (e.g. a BUY after a SHORT) arrives before the planned exit, the trade closes early at that date
- **Black swan detection** — if stock moved >3× its 20-day average daily range during the eval window, flagged inconclusive and excluded
- **MFE/MAE** — Maximum Favorable/Adverse Excursion using intraday High/Low, measures whether the thesis was ever right even if the exit was wrong

**Dashboard:** Overall accuracy banner (directional + exit), metrics by confidence level, metrics by time horizon, return distribution histogram, best/worst signals, full table with MFE/MAE columns, CSV download.

---

### 11 — 🧠 Profitability Engine (`signal_profitability.py`)

Self-improving layer. Loads all BUY/SHORT signals from Supabase, evaluates outcomes, then ranks signal types by profitability.

**Signal categories detected:**
- News LLM (live_intelligence, ticker_signals, market_briefing)
- RSI Mean Reversion, MA Crossover, Momentum, ML Weekly (from strategy_signals)
- Historical Simulation (auto_evaluator)

**Market regime detection:** Computes SPY 50-day MA deviation to classify each signal's date as bull / bear / sideways.

**Per (signal type × regime) metrics:**
- Directional accuracy % (MFE-based, threshold adjustable)
- Exit accuracy % (for reference)
- Avg return %
- Volatility-adjusted return (return ÷ ticker annualised vol — like a per-signal Sharpe)
- Composite Score = directional accuracy × avg return

**Auto-ranking:** ✅ Keep (score > 0.5), ⚠️ Watch (score 0–0.5), ❌ Kill (score < 0).

**Visualisations:** Strategy composite score bar chart, return heatmap (signal type × regime), directional vs exit accuracy comparison by confidence level, cumulative return curves by signal type, volatility-adjusted return chart.

**Plain-English intelligence:** Auto-generated paragraphs like "RSI Mean Reversion is your best performer — directional accuracy 67%, avg return +1.8% in bull markets."

---

### 12 — 📊 Stock Correlations (`correlation_engine.py`)

Downloads price data via yfinance, computes full correlation matrix, then layers on ML prediction.

**Output:**
- Pairwise correlation table (colour-coded: green ≥ 0.7, amber 0.3–0.7, red ≤ -0.3)
- Correlation heatmap (for 3+ tickers)
- Rolling N-day correlation chart (configurable window, up to 6 pairs)
- Random Forest model predicting next-step rolling correlation between the primary pair

**RF model details:**
- Features: all rolling correlation series
- Target: next-day correlation of primary pair
- Walk-forward: trains on first 80% (configurable), tests on last 20%
- Metrics: MSE, R² score, actual vs predicted chart

Results saved to Supabase session state and shared with Portfolio Risk and AI Portfolio Advisor.

---

### 13 — ⚠️ Portfolio Risk (`portfolio_risk.py`)

Synthesises correlation data (from Tab 12) and sentiment data (from Tab 8) into a composite risk score and named risk flags.

**Correlation flags:**
- 🔴 High correlation cluster (≥ 0.80) — reduced diversification
- 🟡 Moderate correlation (0.50–0.79)
- 🟢 Natural hedge detected (≤ -0.50)

**Sentiment flags:**
- 🔴 High hype + low truthfulness
- 🔴 High panic signal
- 🟡 Propaganda detected
- 🟡 Political bias in coverage
- 🟢 Calm news environment

**Combined score formula:**
```
risk_score = (avg_correlation × 0.4) + (avg_hype × 0.3) + (avg_panic × 0.3)
```
Clamped 0–1. Labelled LOW / MODERATE / HIGH with colour-coded banner.

---

### 14 — 🤖 AI Portfolio Advisor (`advisor_engine.py`)

Conversational AI chat grounded in the user's actual computed data — not generic financial advice.

**Context injected into every Groq call:**
- Portfolio tickers
- Full pairwise correlation table with cluster summaries
- Recent 30-day rolling correlation averages with trend direction (↑/↓/→)
- Per-article sentiment scores (hype, panic, truthfulness, propaganda)

**System prompt role:** "FinSight AI — expert financial advisor and quantitative analyst." Instructed to always cite specific correlation values and sentiment scores, give concrete recommendations, and flag risks clearly.

**Response structure:** Correlation Insights → Sentiment Outlook → Risk Assessment → Specific Recommendations → Outlook → Disclaimer.

Full conversation history maintained in session state. Quick-start prompt buttons for common questions. Clear conversation button.

---

### 15 — 📜 Signal History Log (`signal_history.py`)

Every signal from every tab is automatically logged here. The Supabase `signals` table schema:

```sql
CREATE TABLE signals (
  id            BIGSERIAL PRIMARY KEY,
  timestamp     TEXT NOT NULL,
  source        TEXT NOT NULL,      -- live_intelligence | ticker_signals | strategy_signals | etc.
  ticker        TEXT NOT NULL,
  action        TEXT NOT NULL,      -- BUY | SHORT | HOLD | WATCH
  confidence    TEXT NOT NULL,      -- HIGH | MEDIUM | LOW
  urgency       TEXT,
  market_impact TEXT,
  time_horizon  TEXT,
  reasoning     TEXT,
  article_title TEXT,
  article_url   TEXT,
  source_feed   TEXT
);
```

**Dashboard features:** Filter by time period (1–365 days), source, action, and ticker search. Summary stats (total, BUY count, SHORT count, HIGH confidence count, HIGH %). Signal breakdown pie chart, most-signalled tickers bar chart, daily signal activity timeline. Full filterable table. CSV download. Clear all button.

Always-visible Supabase connection diagnostics at the top of the tab — shows exactly what's broken if the connection fails, with fix instructions.

---

### 16 — 📧 Email Alerts (`email_alerts.py`)

Automatic HIGH urgency email alerts from Live News Feed. Supports two methods:

**Gmail SMTP** (easiest): Requires Gmail App Password (not real password). Configure via `.env` or Streamlit Secrets. `EMAIL_METHOD=gmail`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, `ALERT_RECIPIENT`.

**SendGrid API** (more scalable): `EMAIL_METHOD=sendgrid`, `SENDGRID_API_KEY`, `SENDGRID_FROM`, `ALERT_RECIPIENT`.

Email format: styled HTML with dark theme matching the app. Shows article source + title, market impact badge, full recommendations table (ticker, action, confidence, horizon, reasoning), key risk, link to full article, disclaimer.

Only fires when Live Feed generates a HIGH urgency signal AND the alert recipient is configured. Silent otherwise — no errors surfaced to the UI.

---

## Data Flows

### Signal Lifecycle
```
RSS feed / user input
    → Groq analysis
    → Signal card rendered in UI
    → log_signal() → Supabase signals table
    → eval_core evaluates after horizon elapses
    → accuracy metrics → Profitability Engine ranking
```

### Portfolio Data
```
User adds holding → Supabase holdings table
    → Portfolio Tracker loads on each render
    → yfinance fetches live prices + technicals
    → News signals overlaid from session state
    → Risk flags from correlation + sentiment data
```

### Evaluation Engine (`eval_core.py`)
```
Signal timestamp
    → Find next trading day (via SPY calendar)
    → Fetch OPEN price of entry day
    → Determine exit date (horizon OR superseding signal, whichever first)
    → Fetch CLOSE of exit day
    → Compute return (directional: BUY profits from rise, SHORT from fall)
    → Fetch intraday OHLC for MFE/MAE
    → Detect volatility spike (3× normal range → inconclusive)
    → Classify: ready / waiting / inconclusive
```

---

## Database Schema (Supabase)

```sql
-- Signal history (all sources)
CREATE TABLE signals (
  id BIGSERIAL PRIMARY KEY,
  timestamp TEXT NOT NULL,
  source TEXT NOT NULL,
  ticker TEXT NOT NULL,
  action TEXT NOT NULL,
  confidence TEXT NOT NULL,
  urgency TEXT, market_impact TEXT, time_horizon TEXT,
  reasoning TEXT, article_title TEXT, article_url TEXT, source_feed TEXT
);

-- Portfolio holdings
CREATE TABLE holdings (
  id BIGSERIAL PRIMARY KEY,
  ticker TEXT NOT NULL UNIQUE,
  shares REAL NOT NULL,
  avg_cost REAL NOT NULL,
  notes TEXT, added_at TEXT
);
```

**Row Level Security:** Must be disabled on both tables (or explicit anon-key policies added) for the app to write. The `supabase_test.py` diagnostic script walks through every failure mode.

---

## Configuration

### Required
| Secret | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — all AI analysis |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/public key |

### Optional (Email Alerts)
| Secret | Description |
|---|---|
| `EMAIL_METHOD` | `gmail` or `sendgrid` |
| `GMAIL_ADDRESS` | Your Gmail address |
| `GMAIL_APP_PASSWORD` | 16-char Gmail App Password |
| `ALERT_RECIPIENT` | Where to send alerts |
| `SENDGRID_API_KEY` | SendGrid API key (if using SendGrid) |
| `SENDGRID_FROM` | Verified sender address |

---

## Sidebar Controls (Global)

The sidebar controls affect the Correlation Analytics and Strategy Simulator tabs:

| Control | Default | Affects |
|---|---|---|
| Data Start | 2019-01-01 | Correlation download window |
| Rolling Window (days) | 30 | Rolling correlation chart |
| RF Trees | 100 | Correlation predictor model |
| Train Split % | 80 | Correlation predictor train/test |

---

## Daily Workflow (Recommended)

1. **Morning Briefing** — run first, see the AI market overview for the day
2. **Live News Feed** — press Start, leave running all day, signals auto-log
3. **Ticker Signal Lookup** — deep-dive on specific tickers you're tracking
4. **Portfolio Tracker** — check your holdings' P&L, technicals, and news signals
5. **Strategy Backtester** → **Live Trading Signals** — find best params, generate forward signal
6. **Signal Performance / News Signal Accuracy** — after 5+ trading days, evaluate past signals
7. **Profitability Engine** — after 2–3 weeks, see which signal types are actually making money
8. **AI Portfolio Advisor** — run Correlations + Sentiment first, then ask specific questions

---

## What Was Added in This Update

**`strategy_signals.py` — Signal Accuracy Tracker section (new, at the bottom of Tab 6):**

Every BUY/SHORT generated in Live Trading Signals is saved to Supabase. Once those signals have aged past their stated time horizon, the Signal Accuracy Tracker evaluates them:

- Pulls all `source = "strategy_signals"` rows from Supabase
- Runs through `eval_core` (same engine as Signal Performance and News Signal Accuracy)
- Entry = OPEN of next trading day; Exit = CLOSE at horizon; MFE/MAE from intraday High/Low
- Black-swan exclusion (3× normal move during eval window → inconclusive)
- Supersession detection (opposing signal before horizon → early exit)

**Metrics surfaced:**
- Directional accuracy % with adjustable MFE threshold slider
- Exit accuracy % with t-test statistical significance
- Avg return, std dev, variance, rough Sharpe
- Win rate, profit factor, best/worst signal
- Per-strategy breakdown table (with Sharpe, p-value, significance)
- Per-ticker breakdown table
- Variance profile (skewness, kurtosis, coefficient of variation)
- Return distribution chart with normal fit
- Cumulative P&L curve
- Rolling accuracy over time
- Box plot (BUY vs SHORT spread)
- MFE/MAE scatter with "right direction, wrong timing" detection
- Full evaluated signals table with CSV download

---

*FinSight AI · Not financial advice · All signal evaluations use real market prices from Yahoo Finance*

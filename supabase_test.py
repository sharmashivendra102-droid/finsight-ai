"""
supabase_test.py — Drop this in your PROJECT ROOT (same folder as app.py)
Then go to finsight-aiapp.streamlit.app/supabase_test

Run with: streamlit run supabase_test.py

This will tell you EXACTLY what is broken.
Delete this file after you fix the issue.
"""

import streamlit as st
import os

st.set_page_config(page_title="Supabase Diagnostic", page_icon="🔧")
st.title("🔧 Supabase Diagnostic")
st.caption("This will tell you exactly what is wrong. Delete after fixing.")

# ── Step 1: Check secrets ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 1 — Secrets")

url = ""
key = ""

try:
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
except Exception as e:
    st.error(f"❌ Could not read st.secrets at all: {e}")

if url:
    st.success(f"✅ SUPABASE_URL found: `{url[:40]}...`")
else:
    st.error("❌ SUPABASE_URL is missing or empty in Streamlit Secrets")
    st.info("Go to share.streamlit.io → your app → Settings → Secrets and add:\n```\nSUPABASE_URL = \"https://xxxx.supabase.co\"\n```")

if key:
    st.success(f"✅ SUPABASE_KEY found: `{key[:20]}...` (length: {len(key)})")
else:
    st.error("❌ SUPABASE_KEY is missing or empty in Streamlit Secrets")
    st.info("Add the **anon/public** key (starts with eyJ...) from Supabase → Project Settings → API")

if not url or not key:
    st.stop()

# ── Step 2: Check package ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 2 — supabase Package")

try:
    from supabase import create_client
    st.success("✅ supabase package is installed and importable")
except ImportError as e:
    st.error(f"❌ supabase package not installed: {e}")
    st.info("Add `supabase>=2.0.0` to requirements.txt and push to GitHub. "
            "Streamlit Cloud needs to reinstall dependencies.")
    st.stop()

# ── Step 3: Create client ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 3 — Client Connection")

try:
    client = create_client(url, key)
    st.success("✅ Supabase client created successfully")
except Exception as e:
    st.error(f"❌ Could not create Supabase client: {e}")
    st.info("This usually means the URL or key is malformed. "
            "Copy them fresh from Supabase → Project Settings → API.")
    st.stop()

# ── Step 4: Test signals table ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 4 — signals Table")

try:
    resp = client.table("signals").select("id", count="exact").execute()
    count = resp.count if resp.count is not None else 0
    st.success(f"✅ signals table exists and is reachable — {count} rows")
except Exception as e:
    st.error(f"❌ Cannot access signals table: {e}")
    st.info("""Run this SQL in Supabase → SQL Editor:

```sql
CREATE TABLE signals (
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
```
""")
    st.stop()

# ── Step 5: Test holdings table ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 5 — holdings Table")

try:
    resp2 = client.table("holdings").select("id", count="exact").execute()
    count2 = resp2.count if resp2.count is not None else 0
    st.success(f"✅ holdings table exists and is reachable — {count2} rows")
except Exception as e:
    st.error(f"❌ Cannot access holdings table: {e}")
    st.info("""Run this SQL in Supabase → SQL Editor:

```sql
CREATE TABLE holdings (
  id        BIGSERIAL PRIMARY KEY,
  ticker    TEXT NOT NULL UNIQUE,
  shares    REAL NOT NULL,
  avg_cost  REAL NOT NULL,
  notes     TEXT,
  added_at  TEXT
);
```
""")
    st.stop()

# ── Step 6: Test write ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 6 — Write Test")

if st.button("🧪 Write a test signal row"):
    try:
        client.table("signals").insert({
            "timestamp":     "2025-01-01 00:00:00",
            "source":        "diagnostic_test",
            "ticker":        "TEST",
            "action":        "BUY",
            "confidence":    "HIGH",
            "urgency":       "LOW",
            "market_impact": "NEUTRAL",
            "time_horizon":  "INTRADAY",
            "reasoning":     "Supabase diagnostic test row",
            "article_title": "Test",
            "article_url":   "",
            "source_feed":   "test",
        }).execute()
        st.success("✅ Write succeeded! Go check Supabase → Table Editor → signals. "
                   "You should see a row with ticker=TEST.")
        st.balloons()
    except Exception as e:
        st.error(f"❌ Write failed: {e}")
        st.info("This means the table exists but inserts are being rejected. "
                "Check Supabase → Authentication → Policies. "
                "You may need to disable RLS (Row Level Security) for the signals table.")

# ── Step 7: RLS check ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 7 — Row Level Security (RLS)")
st.info("""
**This is the most common hidden issue.**

Supabase enables Row Level Security by default, which blocks all writes
from the anon key unless you explicitly add policies.

Fix: Go to Supabase → **Authentication → Policies**

For each table (signals AND holdings), either:

**Option A (easiest):** Disable RLS entirely
- Click the table → toggle "Enable RLS" off

**Option B (more secure):** Add a policy
- Click "New Policy" → "For full customization"
- Policy name: `allow_all`
- Target roles: `anon`
- USING expression: `true`
- WITH CHECK expression: `true`
- Apply to: SELECT, INSERT, UPDATE, DELETE
""")

st.markdown("---")
st.success("✅ If all 6 steps above passed, Supabase is fully working. "
           "Delete this file and push to GitHub.")

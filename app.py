"""
Supplier Risk Assessor — Tier 2 Steel Car Parts Manufacturing
=============================================================
A free, AI-powered tool that evaluates supplier reliability, predicts
potential delays, and suggests alternative vendors for global steel sourcing.

Tech stack: Streamlit · Python · scikit-learn · Free public APIs
Target users: Tier 2 automotive steel parts manufacturers
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
import sys
import os
from typing import Optional

# ── Ensure local packages are importable on Streamlit Cloud ──────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _APP_DIR)

# Streamlit Cloud clones from git, which can silently drop __init__.py files.
# Create them at runtime if they're missing so Python recognizes the packages.
for _pkg in ("utils", "components"):
    _pkg_dir = os.path.join(_APP_DIR, _pkg)
    _init_file = os.path.join(_pkg_dir, "__init__.py")
    if os.path.isdir(_pkg_dir) and not os.path.exists(_init_file):
        with open(_init_file, "w") as _f:
            _f.write(f"# auto-created for {_pkg}\n")

# ── Local modules ────────────────────────────────────────────────────────────
from utils.api_clients import (
    fetch_world_bank_indicators,
    fetch_comtrade_steel_data,
    fetch_commodity_prices_fred,
    fetch_news_events,
    fetch_exchange_rates,
)
from utils.risk_engine import (
    compute_supplier_risk_score,
    predict_delay_probability,
    generate_risk_breakdown,
    RISK_WEIGHTS,
)
from utils.vendor_recommender import (
    get_alternative_vendors,
    rank_vendors,
    VENDOR_DATABASE,
)
from utils.report_generator import build_report_dataframe, render_report
from components.ui_helpers import (
    render_header,
    render_sidebar_inputs,
    render_risk_gauge,
    render_delay_timeline,
    render_vendor_cards,
    render_buyer_banner,
    severity_color,
    TRADE_AGREEMENT_MEMBERS,
)

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Supplier Risk Assessor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 1.25rem; color: white;
        border-left: 4px solid #e94560; margin-bottom: 0.75rem;
    }
    .metric-card h4 { margin: 0 0 0.5rem 0; font-size: 0.85rem;
                       text-transform: uppercase; letter-spacing: 1px;
                       color: #a0a0b0; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700;
                          font-family: 'JetBrains Mono', monospace; }

    /* Risk badge */
    .risk-badge {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
    }
    .risk-low    { background: #064e3b; color: #6ee7b7; }
    .risk-medium { background: #78350f; color: #fcd34d; }
    .risk-high   { background: #7f1d1d; color: #fca5a5; }
    .risk-critical { background: #450a0a; color: #ff6b6b; }

    /* Vendor card */
    .vendor-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;
        transition: transform 0.15s;
    }
    .vendor-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }

    /* Report section */
    .report-section {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#                              MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()

    # ── Sidebar inputs ───────────────────────────────────────────────────────
    inputs = render_sidebar_inputs()

    if not inputs:
        # Landing / instructions
        st.markdown("---")
        col0, _ = st.columns([2, 1])
        with col0:
            st.markdown("""
            ### 🏢 Start Here — Identify Your Company
            Fill in your company profile in the sidebar so the tool
            can tailor risk scores to **your geography**, compliance
            obligations, and supply-chain position.
            """)
        st.markdown("")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 📋 Step 1 — Enter Suppliers
            Add your current supplier names, locations, and the steel
            materials you source from each.
            """)
        with col2:
            st.markdown("""
            ### 🔍 Step 2 — Set Risk Factors
            Choose the risk categories you care about: geopolitical,
            logistics, financial, environmental, or market volatility.
            """)
        with col3:
            st.markdown("""
            ### 📊 Step 3 — Generate Report
            Click **Assess Risk** to get scores, delay predictions,
            and ranked alternative vendor suggestions.
            """)
        return

    buyer_profile  = inputs["buyer_profile"]
    suppliers      = inputs["suppliers"]
    material_type  = inputs["material_type"]
    risk_concerns  = inputs["risk_concerns"]

    # Show buyer context banner below the header
    render_buyer_banner(buyer_profile)

    # ── Run assessment ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Assessment Results")

    progress = st.progress(0, text="Initializing risk engine…")

    # 1. Fetch macro-economic data ─────────────────────────────────────────
    progress.progress(10, text="Fetching macro-economic indicators…")
    # Include buyer's countries for comparative analysis
    all_country_codes = list(set(
        [s.get("country_code", "US") for s in suppliers]
        + buyer_profile.get("manufacturing_country_codes", [])
        + [buyer_profile.get("hq_country_code", "US")]
    ))
    wb_data = fetch_world_bank_indicators(all_country_codes)

    # 2. Fetch trade / commodity data ──────────────────────────────────────
    progress.progress(25, text="Pulling commodity & trade data…")
    commodity_data = fetch_commodity_prices_fred()
    trade_data = fetch_comtrade_steel_data(
        [s.get("country_code", "USA") for s in suppliers]
    )

    # 3. Fetch news / event signals ────────────────────────────────────────
    progress.progress(40, text="Scanning global events & news…")
    news_signals = fetch_news_events(
        material_type, [s["name"] for s in suppliers]
    )

    # 4. Exchange-rate data ────────────────────────────────────────────────
    progress.progress(50, text="Checking exchange-rate exposure…")
    # Include buyer's operating currency for cross-rate analysis
    all_currencies = list(set(
        [s.get("currency", "USD") for s in suppliers]
        + [buyer_profile.get("operating_currency", "USD")]
    ))
    fx_data = fetch_exchange_rates(all_currencies)

    # 5. Score each supplier ───────────────────────────────────────────────
    progress.progress(65, text="Computing risk scores…")
    results = []
    for supplier in suppliers:
        score, breakdown = compute_supplier_risk_score(
            supplier=supplier,
            wb_data=wb_data,
            commodity_data=commodity_data,
            trade_data=trade_data,
            news_signals=news_signals,
            fx_data=fx_data,
            risk_concerns=risk_concerns,
            buyer_profile=buyer_profile,
        )
        delay_prob, delay_details = predict_delay_probability(
            supplier=supplier,
            risk_score=score,
            commodity_data=commodity_data,
            news_signals=news_signals,
        )
        results.append({
            "supplier": supplier,
            "risk_score": score,
            "breakdown": breakdown,
            "delay_probability": delay_prob,
            "delay_details": delay_details,
        })

    # 6. Find alternative vendors ──────────────────────────────────────────
    progress.progress(80, text="Ranking alternative vendors…")
    alternatives = get_alternative_vendors(
        material_type=material_type,
        current_suppliers=suppliers,
        risk_results=results,
        commodity_data=commodity_data,
        buyer_profile=buyer_profile,
    )
    ranked_alts = rank_vendors(alternatives, material_type, buyer_profile=buyer_profile)

    progress.progress(100, text="Done ✓")

    # ── Display results ──────────────────────────────────────────────────
    # Top-line KPI cards
    avg_risk = np.mean([r["risk_score"] for r in results])
    max_delay = max(r["delay_probability"] for r in results)
    high_risk_count = sum(1 for r in results if r["risk_score"] >= 65)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Average Risk Score</h4>
            <div class="value">{avg_risk:.1f}<span style="font-size:.9rem">/100</span></div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#f59e0b">
            <h4>Peak Delay Probability</h4>
            <div class="value">{max_delay:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#8b5cf6">
            <h4>High-Risk Suppliers</h4>
            <div class="value">{high_risk_count} / {len(suppliers)}</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#06b6d4">
            <h4>Alternatives Found</h4>
            <div class="value">{len(ranked_alts)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Per-supplier detail tabs ─────────────────────────────────────────
    tabs = st.tabs([r["supplier"]["name"] for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            s = result["supplier"]
            col_gauge, col_detail = st.columns([1, 2])

            with col_gauge:
                render_risk_gauge(result["risk_score"])
                badge_class = (
                    "risk-low" if result["risk_score"] < 35 else
                    "risk-medium" if result["risk_score"] < 65 else
                    "risk-high" if result["risk_score"] < 85 else
                    "risk-critical"
                )
                badge_label = (
                    "LOW RISK" if result["risk_score"] < 35 else
                    "MODERATE" if result["risk_score"] < 65 else
                    "HIGH RISK" if result["risk_score"] < 85 else
                    "CRITICAL"
                )
                st.markdown(
                    f'<div style="text-align:center"><span class="risk-badge {badge_class}">'
                    f'{badge_label}</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.metric("Delay Probability", f"{result['delay_probability']:.0f}%")

            with col_detail:
                st.markdown("**Risk Breakdown**")
                breakdown_df = pd.DataFrame(
                    [{"Factor": k, "Score": v, "Weight": RISK_WEIGHTS.get(k, 0)}
                     for k, v in result["breakdown"].items()]
                )
                st.dataframe(
                    breakdown_df.style.background_gradient(
                        subset=["Score"], cmap="RdYlGn_r", vmin=0, vmax=100
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                if result["delay_details"]:
                    st.markdown("**Delay Risk Factors**")
                    for d in result["delay_details"]:
                        icon = "🔴" if d["severity"] == "high" else "🟡" if d["severity"] == "medium" else "🟢"
                        st.markdown(f"{icon} **{d['factor']}** — {d['description']}")

    # ── Alternative vendors ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 Recommended Alternative Vendors")
    st.caption(
        "Ranked by composite score (reliability, proximity, cost, compatibility)"
    )
    render_vendor_cards(ranked_alts[:5])

    # ── Downloadable report ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Export Full Report")
    report_df = build_report_dataframe(results, ranked_alts)
    render_report(report_df, results, ranked_alts, material_type, buyer_profile=buyer_profile)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

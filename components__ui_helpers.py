"""
components/ui_helpers.py
========================
Reusable Streamlit UI components for the Supplier Risk Assessor.
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Any


# ── Country / region reference data ──────────────────────────────────────────
COUNTRY_OPTIONS = {
    "United States": "US", "Canada": "CA", "Mexico": "MX",
    "Germany": "DE", "France": "FR", "Italy": "IT", "United Kingdom": "GB",
    "Poland": "PL", "Czech Republic": "CZ", "Slovakia": "SK",
    "Austria": "DE", "Spain": "DE",
    "Japan": "JP", "South Korea": "KR", "China": "CN", "Taiwan": "TW",
    "India": "IN", "Vietnam": "VN", "Thailand": "TH", "Indonesia": "ID",
    "Turkey": "TR", "Brazil": "BR", "Argentina": "AR",
    "Australia": "AU", "South Africa": "ZA",
    "Saudi Arabia": "SA", "UAE": "AE", "Egypt": "EG",
    "Russia": "RU", "Ukraine": "UA",
}

CURRENCY_MAP = {
    "US": "USD", "CA": "CAD", "MX": "MXN", "DE": "EUR", "FR": "EUR",
    "IT": "EUR", "GB": "GBP", "PL": "PLN", "CZ": "CZK", "SK": "EUR",
    "JP": "JPY", "KR": "KRW", "CN": "CNY", "TW": "TWD",
    "IN": "INR", "VN": "VND", "TH": "THB", "ID": "IDR",
    "TR": "TRY", "BR": "BRL", "AR": "ARS",
    "AU": "AUD", "ZA": "ZAR", "SA": "SAR", "AE": "AED",
    "RU": "RUB", "UA": "UAH", "EG": "EGP",
}

MATERIAL_OPTIONS = [
    "high-strength steel coils",
    "cold-rolled steel sheets",
    "galvanized steel",
    "AHSS (Advanced High-Strength)",
    "hot-rolled steel coils",
    "press-hardened steel",
    "stainless steel",
    "electrical steel",
    "alloy steel",
]

RISK_CONCERN_OPTIONS = [
    "Geopolitical risks",
    "Shipping / logistics",
    "Price volatility",
    "Single-source dependency",
    "Currency fluctuations",
    "Climate / natural disasters",
    "Tariffs / regulations",
    "Financial instability",
]

# ── Buyer / company profile reference data ───────────────────────────────────
COMPANY_SIZE_OPTIONS = [
    "Small (< 5,000 MT/year)",
    "Medium (5,000–50,000 MT/year)",
    "Large (50,000–200,000 MT/year)",
    "Enterprise (> 200,000 MT/year)",
]

COMPLIANCE_FRAMEWORKS = [
    "USMCA / CUSMA (North America)",
    "EU CBAM (Carbon Border Adjustment)",
    "EU REACH (Chemical Safety)",
    "EU RoHS (Hazardous Substances)",
    "Buy America / Buy American Act",
    "Section 232 Steel Tariffs (US)",
    "UK REACH (post-Brexit)",
    "CPTPP (Trans-Pacific)",
    "India BIS Certification",
    "ISO 14001 (Environmental)",
    "ISO 45001 (Occupational H&S)",
    "Conflict Minerals (Dodd-Frank 1502)",
    "CBAM / ETS (Emissions Trading)",
]

CERTIFICATION_REQUIREMENTS = [
    "IATF 16949 (Automotive QMS)",
    "ISO 9001 (Quality Management)",
    "ISO 14001 (Environmental)",
    "ISO 50001 (Energy Management)",
    "ISO 45001 (Health & Safety)",
    "AS9100 (Aerospace — if dual-use)",
    "NADCAP (Special Processes)",
]

AUTOMOTIVE_OEM_TIERS = [
    "Tier 1 (direct to OEM)",
    "Tier 2 (supply to Tier 1)",
    "Tier 3 (sub-component / raw material)",
    "Tier 2+ (mixed Tier 1 & Tier 2)",
]

# ── Trade agreement zones (for compliance matching) ──────────────────────────
TRADE_AGREEMENT_MEMBERS = {
    "USMCA / CUSMA (North America)": ["US", "CA", "MX"],
    "EU CBAM (Carbon Border Adjustment)": [
        "DE", "FR", "IT", "GB", "PL", "CZ", "SK",
    ],
    "EU REACH (Chemical Safety)": [
        "DE", "FR", "IT", "GB", "PL", "CZ", "SK",
    ],
    "EU RoHS (Hazardous Substances)": [
        "DE", "FR", "IT", "GB", "PL", "CZ", "SK",
    ],
    "Buy America / Buy American Act": ["US"],
    "Section 232 Steel Tariffs (US)": ["US"],
    "UK REACH (post-Brexit)": ["GB"],
    "CPTPP (Trans-Pacific)": [
        "JP", "AU", "CA", "MX", "VN", "GB",
    ],
    "India BIS Certification": ["IN"],
}

REGION_MAP = {
    "US": "North America", "CA": "North America", "MX": "North America",
    "DE": "Western Europe", "FR": "Western Europe", "IT": "Western Europe",
    "GB": "Western Europe", "PL": "Eastern Europe", "CZ": "Eastern Europe",
    "SK": "Eastern Europe",
    "JP": "East Asia", "KR": "East Asia", "CN": "East Asia", "TW": "East Asia",
    "IN": "South Asia", "VN": "Southeast Asia", "TH": "Southeast Asia",
    "ID": "Southeast Asia",
    "TR": "Middle East", "SA": "Middle East", "AE": "Middle East",
    "BR": "South America", "AR": "South America",
    "AU": "Oceania", "ZA": "Africa", "EG": "Africa",
    "RU": "Eastern Europe", "UA": "Eastern Europe",
}


def render_header(buyer_profile: Optional[Dict] = None):
    """Render the app header."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #334155 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        border-bottom: 3px solid #e94560;
    ">
        <h1 style="color: white; margin: 0; font-size: 2rem;">
            🏭 Steel Supplier Risk Assessor
        </h1>
        <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.05rem;">
            Tier 2 Automotive Parts Manufacturing — Global Supply Chain Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_buyer_banner(buyer_profile: Dict):
    """
    Render the buyer identity context banner below the header.
    Shows who is purchasing, their compliance scope, and volume.
    """
    bp = buyer_profile
    if not bp or not bp.get("company_name"):
        return

    sites_str = ", ".join(bp.get("manufacturing_countries", []))
    compliance_short = ", ".join(
        c.split("(")[0].strip() for c in bp.get("compliance_frameworks", [])[:3]
    )
    if len(bp.get("compliance_frameworks", [])) > 3:
        compliance_short += f" +{len(bp['compliance_frameworks']) - 3} more"

    cert_short = ", ".join(
        c.split("(")[0].strip() for c in bp.get("required_certifications", [])[:3]
    )

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0c1426 0%, #162032 100%);
        border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 1rem 1.5rem; margin-bottom: 1rem;
        display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;
    ">
        <div>
            <div style="font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 1px; color: #64748b; margin-bottom: 4px;">
                Purchasing Company
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0;">
                {bp['company_name']}
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                {bp.get('tier_position', 'Tier 2')} · {bp.get('hq_country', '')}
                {(' · Mfg: ' + sites_str) if sites_str else ''}
            </div>
        </div>
        <div>
            <div style="font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 1px; color: #64748b; margin-bottom: 4px;">
                Compliance Requirements
            </div>
            <div style="font-size: 0.85rem; color: #93c5fd;">
                {compliance_short or 'None specified'}
            </div>
        </div>
        <div>
            <div style="font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 1px; color: #64748b; margin-bottom: 4px;">
                Volume & Certifications
            </div>
            <div style="font-size: 0.85rem; color: #a5b4fc;">
                {bp.get('annual_volume', 'Not specified')}
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                Requires: {cert_short or 'None specified'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_inputs() -> Optional[Dict]:
    """
    Render the sidebar input form. Returns dict of inputs when
    the user clicks Assess Risk, or None if not submitted.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # ═══════════════════════════════════════════════════════════════════
        #  SECTION 1: BUYER / COMPANY PROFILE  (who is purchasing?)
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🏢 Your Company Profile")
        st.caption(
            "Identifies who is purchasing so the assessment is tailored to "
            "your geography, compliance obligations, and supply-chain position."
        )

        company_name = st.text_input(
            "Company name *",
            value="Midwest Precision Stampings, Inc.",
            key="buyer_company_name",
            help="Legal entity name that will appear on reports.",
        )

        hq_country = st.selectbox(
            "Headquarters country *",
            options=list(COUNTRY_OPTIONS.keys()),
            index=0,  # United States
            key="buyer_hq_country",
        )
        hq_cc = COUNTRY_OPTIONS[hq_country]

        mfg_countries = st.multiselect(
            "Manufacturing site countries",
            options=list(COUNTRY_OPTIONS.keys()),
            default=["United States", "Mexico"],
            key="buyer_mfg_countries",
            help="All countries where you operate stamping/forming plants.",
        )
        mfg_ccs = [COUNTRY_OPTIONS[c] for c in mfg_countries]

        tier_position = st.selectbox(
            "Supply chain position",
            options=AUTOMOTIVE_OEM_TIERS,
            index=1,  # Tier 2
            key="buyer_tier",
        )

        annual_volume = st.selectbox(
            "Annual steel purchase volume",
            options=COMPANY_SIZE_OPTIONS,
            index=1,  # Medium
            key="buyer_volume",
        )

        operating_currency = st.selectbox(
            "Primary operating currency",
            options=["USD", "EUR", "GBP", "JPY", "CAD", "MXN", "CNY", "KRW",
                     "INR", "BRL", "TRY", "AUD", "PLN", "CZK", "THB"],
            index=0,
            key="buyer_currency",
        )

        with st.expander("Compliance & Certifications", expanded=False):
            compliance_frameworks = st.multiselect(
                "Applicable compliance frameworks",
                options=COMPLIANCE_FRAMEWORKS,
                default=["USMCA / CUSMA (North America)",
                         "Section 232 Steel Tariffs (US)"],
                key="buyer_compliance",
                help="Trade agreements and regulations that constrain your sourcing.",
            )
            required_certs = st.multiselect(
                "Required supplier certifications",
                options=CERTIFICATION_REQUIREMENTS,
                default=["IATF 16949 (Automotive QMS)",
                         "ISO 9001 (Quality Management)"],
                key="buyer_req_certs",
                help="Certifications suppliers MUST hold to qualify.",
            )
            oem_customers = st.text_input(
                "Key OEM customers (optional)",
                value="",
                key="buyer_oems",
                help="e.g. 'Ford, GM, Stellantis' — helps tailor compliance checks.",
            )

        # Build the buyer profile dict
        buyer_profile = {
            "company_name": company_name,
            "hq_country": hq_country,
            "hq_country_code": hq_cc,
            "hq_region": REGION_MAP.get(hq_cc, "North America"),
            "manufacturing_countries": mfg_countries,
            "manufacturing_country_codes": mfg_ccs,
            "manufacturing_regions": list({
                REGION_MAP.get(c, "North America") for c in mfg_ccs
            }),
            "tier_position": tier_position,
            "annual_volume": annual_volume,
            "operating_currency": operating_currency,
            "compliance_frameworks": compliance_frameworks,
            "required_certifications": required_certs,
            "oem_customers": [
                s.strip() for s in oem_customers.split(",") if s.strip()
            ],
        }

        # ═══════════════════════════════════════════════════════════════════
        #  SECTION 2: SUPPLIER DETAILS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("---")

        # ── Number of suppliers ──────────────────────────────────────────────
        num_suppliers = st.number_input(
            "Number of suppliers to assess",
            min_value=1, max_value=10, value=3, step=1,
        )

        st.markdown("### 📦 Supplier Details")

        suppliers = []
        for i in range(int(num_suppliers)):
            with st.expander(f"Supplier #{i+1}", expanded=(i == 0)):
                name = st.text_input(
                    "Company name",
                    value=_default_supplier_names()[i] if i < 3 else "",
                    key=f"name_{i}",
                )
                country = st.selectbox(
                    "Country",
                    options=list(COUNTRY_OPTIONS.keys()),
                    index=_default_country_idx(i),
                    key=f"country_{i}",
                )
                cc = COUNTRY_OPTIONS[country]

                on_time = st.slider(
                    "On-time delivery rate (%)",
                    min_value=50, max_value=100, value=_default_otd(i),
                    key=f"otd_{i}",
                )
                years = st.number_input(
                    "Years in business",
                    min_value=1, max_value=200, value=_default_years(i),
                    key=f"years_{i}",
                )
                lead_time = st.number_input(
                    "Typical lead time (days)",
                    min_value=1, max_value=180, value=_default_lead(i),
                    key=f"lead_{i}",
                )
                sole_source = st.checkbox(
                    "Sole-source supplier?",
                    value=False,
                    key=f"sole_{i}",
                )

                suppliers.append({
                    "name": name or f"Supplier {i+1}",
                    "country": country,
                    "country_code": cc,
                    "currency": CURRENCY_MAP.get(cc, "USD"),
                    "region": REGION_MAP.get(cc, "North America"),
                    "on_time_delivery_pct": on_time,
                    "years_in_business": years,
                    "lead_time_days": lead_time,
                    "sole_source": sole_source,
                    "alternative_sources": 0 if sole_source else 2,
                })

        st.markdown("---")
        st.markdown("### 🔩 Material Type")
        material_type = st.selectbox(
            "Primary steel product",
            options=MATERIAL_OPTIONS,
            index=0,
        )
        # Store on each supplier for report
        for s in suppliers:
            s["material_type"] = material_type

        st.markdown("---")
        st.markdown("### ⚠️ Risk Priorities")
        risk_concerns = st.multiselect(
            "Select your top concerns",
            options=RISK_CONCERN_OPTIONS,
            default=["Geopolitical risks", "Price volatility", "Shipping / logistics"],
        )

        st.markdown("---")

        submitted = st.button(
            "🔍  Assess Risk",
            use_container_width=True,
            type="primary",
        )

        if submitted:
            return {
                "buyer_profile": buyer_profile,
                "suppliers": suppliers,
                "material_type": material_type,
                "risk_concerns": risk_concerns,
            }

        # API key configuration
        st.markdown("---")
        st.markdown("### 🔑 API Keys (Optional)")
        st.caption(
            "Add free API keys for live data. "
            "The tool works without them using curated data."
        )
        fred_key = st.text_input("FRED API Key", type="password", key="fred_key_input")
        gnews_key = st.text_input("GNews API Key", type="password", key="gnews_key_input")

        if fred_key:
            import os
            os.environ["FRED_API_KEY"] = fred_key
        if gnews_key:
            import os
            os.environ["GNEWS_API_KEY"] = gnews_key

    return None


def render_risk_gauge(score: float):
    """Render a visual risk gauge using HTML/CSS."""
    color = severity_color(score)
    rotation = (score / 100) * 180 - 90  # -90° to 90°

    st.markdown(f"""
    <div style="text-align:center; margin: 1rem 0;">
        <svg width="180" height="110" viewBox="0 0 200 120">
            <!-- Background arc -->
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none"
                  stroke="#e2e8f0" stroke-width="14" stroke-linecap="round"/>
            <!-- Green zone -->
            <path d="M 20 100 A 80 80 0 0 1 60 32" fill="none"
                  stroke="#22c55e" stroke-width="14" stroke-linecap="round"/>
            <!-- Yellow zone -->
            <path d="M 60 32 A 80 80 0 0 1 140 32" fill="none"
                  stroke="#eab308" stroke-width="14" stroke-linecap="round"/>
            <!-- Red zone -->
            <path d="M 140 32 A 80 80 0 0 1 180 100" fill="none"
                  stroke="#ef4444" stroke-width="14" stroke-linecap="round"/>
            <!-- Needle -->
            <line x1="100" y1="100" x2="100" y2="30"
                  stroke="{color}" stroke-width="3" stroke-linecap="round"
                  transform="rotate({rotation}, 100, 100)"/>
            <circle cx="100" cy="100" r="6" fill="{color}"/>
            <!-- Score text -->
            <text x="100" y="95" text-anchor="middle"
                  font-family="JetBrains Mono, monospace" font-size="28"
                  font-weight="bold" fill="{color}">{score:.0f}</text>
            <text x="100" y="115" text-anchor="middle"
                  font-family="DM Sans, sans-serif" font-size="11"
                  fill="#64748b">/ 100</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)


def render_delay_timeline(delay_prob: float, details: List[Dict]):
    """Show a visual delay probability indicator."""
    color = severity_color(delay_prob)
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="background: #f1f5f9; border-radius: 8px; height: 24px;
                     overflow: hidden;">
            <div style="background: {color}; height: 100%; width: {delay_prob}%;
                        border-radius: 8px; transition: width 0.3s;">
            </div>
        </div>
        <p style="text-align: center; margin-top: 0.5rem; font-weight: 600;">
            {delay_prob:.0f}% delay probability (next 90 days)
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_vendor_cards(vendors: List[Dict]):
    """Render ranked vendor recommendation cards."""
    for v in vendors:
        rank = v.get("rank", 0)
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        color = severity_color(100 - v.get("composite_score", 50))  # invert: high score = good

        cert_badges = " · ".join(v.get("certifications", []))
        specialties = ", ".join(v.get("specialties", [])[:4])

        col1, col2, col3 = st.columns([0.6, 2.4, 1])

        with col1:
            st.markdown(f"""
            <div style="text-align:center; padding-top: 0.5rem;">
                <span style="font-size: 2rem;">{medal}</span>
                <div style="font-family: 'JetBrains Mono'; font-size: 1.4rem;
                            font-weight: 700; color: #1e293b;">
                    {v.get('composite_score', 0):.0f}
                </div>
                <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase;">
                    composite
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="vendor-card">
                <div style="display:flex; justify-content:space-between; align-items:start;">
                    <div>
                        <h4 style="margin:0; font-size:1.1rem;">{v['name']}</h4>
                        <p style="margin:0.25rem 0; color:#64748b; font-size:0.85rem;">
                            📍 {v.get('city', '')} · {v.get('country', '')}
                        </p>
                    </div>
                    <span style="background:#f0fdf4; color:#16a34a; padding:2px 8px;
                                 border-radius:12px; font-size:0.75rem; font-weight:600;">
                        {v.get('sustainability_rating', 'N/A')}
                    </span>
                </div>
                <p style="margin:0.5rem 0 0.25rem 0; font-size:0.82rem; color:#475569;">
                    <strong>Specialties:</strong> {specialties}
                </p>
                <p style="margin:0; font-size:0.78rem; color:#64748b;">
                    {cert_badges}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            cert_score = v.get('cert_compliance_score', None)
            trade_score = v.get('trade_agreement_score', None)
            extra_lines = ""
            if cert_score is not None and cert_score != 50:
                cert_color = "#22c55e" if cert_score >= 75 else "#eab308" if cert_score >= 40 else "#ef4444"
                extra_lines += f'<div>📜 Cert match: <strong style="color:{cert_color}">{cert_score:.0f}</strong></div>'
            if trade_score is not None and trade_score != 50:
                trade_color = "#22c55e" if trade_score >= 75 else "#eab308" if trade_score >= 40 else "#ef4444"
                extra_lines += f'<div>🤝 Trade align: <strong style="color:{trade_color}">{trade_score:.0f}</strong></div>'

            st.markdown(f"""
            <div style="font-size: 0.82rem; padding-top: 0.5rem;">
                <div>🎯 Reliability: <strong>{v.get('reliability_score', 0)}</strong></div>
                <div>🔧 Compat: <strong>{v.get('compatibility_score', 0):.0f}</strong></div>
                <div>💰 Cost: <strong>{v.get('cost_score', 0):.0f}</strong></div>
                <div>📍 Proximity: <strong>{v.get('proximity_score', 0):.0f}</strong></div>
                <div>⏱️ Lead: <strong>{v.get('lead_time_days', 0)}d</strong></div>
                {extra_lines}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")


def severity_color(score: float) -> str:
    """Return a CSS color for a 0–100 risk score."""
    if score < 35:
        return "#22c55e"
    elif score < 65:
        return "#eab308"
    elif score < 85:
        return "#f97316"
    return "#ef4444"


# ── Default values for demo ─────────────────────────────────────────────────
def _default_supplier_names():
    return [
        "Shanghai Baogang Group",
        "Hyundai Steel Co.",
        "ArcelorMittal Dofasco",
    ]

def _default_country_idx(i):
    idxs = [11, 5, 1]  # China, South Korea, Canada
    return idxs[i] if i < len(idxs) else 0

def _default_otd(i):
    vals = [82, 91, 88]
    return vals[i] if i < len(vals) else 85

def _default_years(i):
    vals = [30, 71, 112]
    return vals[i] if i < len(vals) else 15

def _default_lead(i):
    vals = [42, 35, 18]
    return vals[i] if i < len(vals) else 30

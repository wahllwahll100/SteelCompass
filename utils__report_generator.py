"""
utils/report_generator.py
=========================
Generate downloadable reports from risk assessment results.
Outputs CSV and a formatted text summary.
"""

import io
import json
import datetime
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional


def build_report_dataframe(
    results: List[Dict],
    alternatives: List[Dict],
) -> pd.DataFrame:
    """Build a consolidated DataFrame for the full assessment."""
    rows = []
    for r in results:
        s = r["supplier"]
        row = {
            "Supplier": s["name"],
            "Country": s.get("country", "N/A"),
            "Material": s.get("material_type", "N/A"),
            "Overall Risk Score": r["risk_score"],
            "Risk Level": _risk_label(r["risk_score"]),
            "Delay Probability (%)": r["delay_probability"],
        }
        # Add breakdown columns
        for factor, score in r["breakdown"].items():
            row[f"  {factor}"] = round(score, 1)
        rows.append(row)

    return pd.DataFrame(rows)


def _risk_label(score: float) -> str:
    if score < 35:
        return "LOW"
    elif score < 65:
        return "MODERATE"
    elif score < 85:
        return "HIGH"
    return "CRITICAL"


def render_report(
    report_df: pd.DataFrame,
    results: List[Dict],
    alternatives: List[Dict],
    material_type: str,
    buyer_profile: Optional[Dict] = None,
):
    """Render download buttons and preview for the full report."""

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    bp = buyer_profile or {}
    company_slug = (
        bp.get("company_name", "assessment").replace(" ", "_")
        .replace(",", "").replace(".", "")[:30].lower()
    )

    col1, col2 = st.columns(2)

    # ── CSV export ───────────────────────────────────────────────────────────
    with col1:
        csv_buf = io.StringIO()

        # Buyer identity header
        csv_buf.write("--- PURCHASING COMPANY ---\n")
        csv_buf.write(f"Company,{bp.get('company_name', 'N/A')}\n")
        csv_buf.write(f"HQ Country,{bp.get('hq_country', 'N/A')}\n")
        csv_buf.write(f"Mfg Sites,\"{', '.join(bp.get('manufacturing_countries', []))}\"\n")
        csv_buf.write(f"Tier Position,{bp.get('tier_position', 'N/A')}\n")
        csv_buf.write(f"Annual Volume,{bp.get('annual_volume', 'N/A')}\n")
        csv_buf.write(f"Operating Currency,{bp.get('operating_currency', 'N/A')}\n")
        csv_buf.write(f"Compliance,\"{', '.join(bp.get('compliance_frameworks', []))}\"\n")
        csv_buf.write(f"Required Certs,\"{', '.join(bp.get('required_certifications', []))}\"\n")
        csv_buf.write(f"Generated,{now}\n\n")

        csv_buf.write("--- SUPPLIER RISK ASSESSMENT ---\n\n")
        report_df.to_csv(csv_buf, index=False)

        # Add alternatives section
        if alternatives:
            csv_buf.write("\n\n--- RECOMMENDED ALTERNATIVE VENDORS ---\n\n")
            alt_df = pd.DataFrame([
                {
                    "Rank": a.get("rank", "-"),
                    "Vendor": a["name"],
                    "Country": a.get("country", ""),
                    "Composite Score": a.get("composite_score", 0),
                    "Reliability": a.get("reliability_score", 0),
                    "Compatibility": a.get("compatibility_score", 0),
                    "Cost Score": a.get("cost_score", 0),
                    "Cert Compliance": a.get("cert_compliance_score", "N/A"),
                    "Trade Agreement": a.get("trade_agreement_score", "N/A"),
                    "Lead Time (days)": a.get("lead_time_days", 0),
                    "Certifications": ", ".join(a.get("certifications", [])),
                }
                for a in alternatives[:5]
            ])
            alt_df.to_csv(csv_buf, index=False)

        st.download_button(
            label="📄 Download CSV Report",
            data=csv_buf.getvalue(),
            file_name=f"risk_report_{company_slug}_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── JSON export ──────────────────────────────────────────────────────────
    with col2:
        report_json = {
            "report_metadata": {
                "generated_at": now,
                "material_type": material_type,
                "tool": "Steel Supplier Risk Assessor v1.0",
            },
            "purchasing_company": {
                "company_name": bp.get("company_name", ""),
                "hq_country": bp.get("hq_country", ""),
                "hq_country_code": bp.get("hq_country_code", ""),
                "manufacturing_countries": bp.get("manufacturing_countries", []),
                "manufacturing_country_codes": bp.get("manufacturing_country_codes", []),
                "tier_position": bp.get("tier_position", ""),
                "annual_volume": bp.get("annual_volume", ""),
                "operating_currency": bp.get("operating_currency", ""),
                "compliance_frameworks": bp.get("compliance_frameworks", []),
                "required_certifications": bp.get("required_certifications", []),
                "oem_customers": bp.get("oem_customers", []),
            },
            "supplier_assessments": [
                {
                    "supplier": r["supplier"]["name"],
                    "country": r["supplier"].get("country", ""),
                    "risk_score": r["risk_score"],
                    "risk_level": _risk_label(r["risk_score"]),
                    "delay_probability_pct": r["delay_probability"],
                    "risk_breakdown": {
                        k: round(v, 1) for k, v in r["breakdown"].items()
                    },
                    "delay_factors": r["delay_details"],
                }
                for r in results
            ],
            "alternative_vendors": [
                {
                    "rank": a.get("rank", 0),
                    "name": a["name"],
                    "country": a.get("country", ""),
                    "composite_score": a.get("composite_score", 0),
                    "reliability_score": a.get("reliability_score", 0),
                    "compatibility_score": a.get("compatibility_score", 0),
                    "cost_score": a.get("cost_score", 0),
                    "cert_compliance_score": a.get("cert_compliance_score", 0),
                    "trade_agreement_score": a.get("trade_agreement_score", 0),
                    "lead_time_days": a.get("lead_time_days", 0),
                    "certifications": a.get("certifications", []),
                    "specialties": a.get("specialties", []),
                }
                for a in alternatives[:5]
            ],
        }

        st.download_button(
            label="📋 Download JSON Report",
            data=json.dumps(report_json, indent=2),
            file_name=f"risk_report_{company_slug}_{datetime.date.today()}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Preview table ────────────────────────────────────────────────────────
    with st.expander("Preview Report Data", expanded=False):
        st.markdown("**Supplier Risk Summary**")
        st.dataframe(
            report_df.style.background_gradient(
                subset=["Overall Risk Score"], cmap="RdYlGn_r", vmin=0, vmax=100
            ),
            use_container_width=True,
            hide_index=True,
        )

        if alternatives:
            st.markdown("**Top 5 Alternative Vendors**")
            alt_preview = pd.DataFrame([
                {
                    "#": a.get("rank", "-"),
                    "Vendor": a["name"],
                    "Country": a.get("country", ""),
                    "Score": a.get("composite_score", 0),
                    "Reliability": a.get("reliability_score", 0),
                    "Compat.": a.get("compatibility_score", 0),
                    "Cost": a.get("cost_score", 0),
                }
                for a in alternatives[:5]
            ])
            st.dataframe(alt_preview, use_container_width=True, hide_index=True)

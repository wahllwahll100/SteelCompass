"""
utils/risk_engine.py
====================
Multi-factor risk scoring engine for steel supplier assessment.

Uses a hybrid approach:
1. Rule-based weighted scoring for transparent, auditable results
2. scikit-learn Random Forest for delay prediction using synthetic
   training data derived from historical supply-chain patterns

Risk score: 0 (safe) → 100 (critical)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ── Risk factor weights (sum = 1.0) ─────────────────────────────────────────
RISK_WEIGHTS = {
    "Geopolitical Stability":     0.18,
    "Financial Health":           0.15,
    "Logistics Reliability":      0.16,
    "Commodity Price Volatility": 0.14,
    "Supply Concentration":       0.12,
    "Currency Exposure":          0.08,
    "Environmental / Climate":    0.07,
    "Regulatory / Tariff Risk":   0.10,
}

# ── Country risk baseline scores (0–100, higher = riskier) ──────────────────
COUNTRY_GEOPOLITICAL_RISK = {
    "US": 15, "CA": 12, "MX": 35, "BR": 40, "DE": 10, "FR": 15,
    "IT": 20, "GB": 12, "JP": 10, "KR": 18, "CN": 45, "IN": 38,
    "TW": 55, "RU": 85, "UA": 90, "TR": 50, "VN": 32, "TH": 28,
    "ID": 33, "AU": 8, "ZA": 42, "SA": 40, "AE": 20, "PL": 18,
    "CZ": 15, "SK": 18, "HU": 25, "RO": 28, "AR": 55, "EG": 48,
}

# ── Logistics baseline by region ────────────────────────────────────────────
REGION_LOGISTICS_RISK = {
    "North America": 15, "Western Europe": 12, "East Asia": 22,
    "South Asia": 45, "Southeast Asia": 35, "Eastern Europe": 30,
    "Middle East": 38, "South America": 42, "Africa": 55,
    "Oceania": 18, "Central America": 40,
}


def _get_region(country_code: str) -> str:
    """Map country code to broad supply-chain region."""
    mapping = {
        "US": "North America", "CA": "North America", "MX": "North America",
        "BR": "South America", "AR": "South America",
        "DE": "Western Europe", "FR": "Western Europe", "IT": "Western Europe",
        "GB": "Western Europe", "PL": "Eastern Europe", "CZ": "Eastern Europe",
        "SK": "Eastern Europe", "HU": "Eastern Europe", "RO": "Eastern Europe",
        "UA": "Eastern Europe", "RU": "Eastern Europe", "TR": "Middle East",
        "JP": "East Asia", "KR": "East Asia", "CN": "East Asia", "TW": "East Asia",
        "IN": "South Asia", "VN": "Southeast Asia", "TH": "Southeast Asia",
        "ID": "Southeast Asia", "AU": "Oceania", "ZA": "Africa",
        "SA": "Middle East", "AE": "Middle East", "EG": "Africa",
    }
    return mapping.get(country_code, "North America")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_supplier_risk_score(
    supplier: Dict,
    wb_data: Dict,
    commodity_data: Dict,
    trade_data: Dict,
    news_signals: List[Dict],
    fx_data: Dict,
    risk_concerns: List[str],
    buyer_profile: Optional[Dict] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a 0–100 risk score for a single supplier.
    When buyer_profile is provided, scores are adjusted for:
      - Trade agreement alignment (USMCA, CPTPP, etc.)
      - Certification compliance gaps
      - Currency cross-rate vs buyer's operating currency
      - Geographic proximity to buyer's manufacturing sites

    Returns (overall_score, {factor: sub_score})
    """
    cc = supplier.get("country_code", "US")
    country_wb = wb_data.get(cc, {})
    region = _get_region(cc)
    bp = buyer_profile or {}

    breakdown: Dict[str, float] = {}

    # 1. Geopolitical Stability ───────────────────────────────────────────────
    geo_base = COUNTRY_GEOPOLITICAL_RISK.get(cc, 30)
    gov_eff = country_wb.get("Government Effectiveness", 0.5)
    # Scale: WB gov effectiveness is roughly -2.5 to 2.5; map to modifier
    geo_modifier = max(-20, min(20, -gov_eff * 10))
    breakdown["Geopolitical Stability"] = np.clip(geo_base + geo_modifier, 0, 100)

    # 2. Financial Health ─────────────────────────────────────────────────────
    gdp_growth = country_wb.get("GDP Growth (%)", 2.5)
    inflation = country_wb.get("Inflation Rate (%)", 4.0)
    supplier_years = supplier.get("years_in_business", 10)
    fin_score = 50  # baseline
    fin_score -= min(20, max(-20, gdp_growth * 4))   # growth reduces risk
    fin_score += min(25, max(0, (inflation - 3) * 5))  # high inflation = risk
    fin_score -= min(15, supplier_years * 1.5)          # tenure reduces risk
    breakdown["Financial Health"] = np.clip(fin_score, 0, 100)

    # 3. Logistics Reliability ────────────────────────────────────────────────
    lpi = country_wb.get("Logistics Performance Index", 3.0)
    logistics_base = REGION_LOGISTICS_RISK.get(region, 30)
    lpi_modifier = (3.0 - lpi) * 15  # below avg LPI increases risk
    delivery_hist = supplier.get("on_time_delivery_pct", 85)
    delivery_penalty = max(0, (90 - delivery_hist) * 1.5)
    # Buyer-aware: if supplier is in the same region as buyer's plants,
    # logistics risk is lower (shorter routes, fewer border crossings)
    buyer_regions = bp.get("manufacturing_regions", [])
    if buyer_regions and region in buyer_regions:
        logistics_base *= 0.7  # 30% reduction for co-located supply
    breakdown["Logistics Reliability"] = np.clip(
        logistics_base + lpi_modifier + delivery_penalty, 0, 100
    )

    # 4. Commodity Price Volatility ───────────────────────────────────────────
    steel_ppi = commodity_data.get("PPI — Steel Mill Products", {})
    pct_chg = abs(steel_ppi.get("pct_change_6m", 0))
    trend = steel_ppi.get("trend", "stable")
    vol_score = min(100, pct_chg * 5)
    if trend == "rising":
        vol_score += 15
    breakdown["Commodity Price Volatility"] = np.clip(vol_score, 0, 100)

    # 5. Supply Concentration ─────────────────────────────────────────────────
    is_sole_source = supplier.get("sole_source", False)
    num_alt_sources = supplier.get("alternative_sources", 2)
    conc_score = 80 if is_sole_source else max(10, 60 - num_alt_sources * 15)
    breakdown["Supply Concentration"] = np.clip(conc_score, 0, 100)

    # 6. Currency Exposure ────────────────────────────────────────────────────
    #    Now measured as cross-rate volatility between supplier's currency
    #    and the BUYER's operating currency (not just USD)
    supplier_currency = supplier.get("currency", "USD")
    buyer_currency = bp.get("operating_currency", "USD")
    if supplier_currency == buyer_currency:
        fx_vol = 0  # same currency = no exposure
    else:
        fx = fx_data.get(supplier_currency, {"volatility_90d": 0})
        fx_vol = fx.get("volatility_90d", 0)
        # If buyer is also non-USD, add a small cross-rate premium
        if buyer_currency != "USD" and supplier_currency != "USD":
            fx_vol *= 1.2
    breakdown["Currency Exposure"] = np.clip(fx_vol * 20, 0, 100)

    # 7. Environmental / Climate ──────────────────────────────────────────────
    climate_risk_map = {
        "Southeast Asia": 55, "South Asia": 60, "East Asia": 35,
        "South America": 40, "Africa": 50, "Middle East": 30,
        "North America": 20, "Western Europe": 15, "Eastern Europe": 22,
        "Oceania": 30, "Central America": 50,
    }
    breakdown["Environmental / Climate"] = climate_risk_map.get(region, 25)

    # 8. Regulatory / Tariff Risk ─────────────────────────────────────────────
    #    Enhanced with buyer-specific compliance framework awareness
    tariff_countries = {"CN": 65, "RU": 80, "IN": 35, "TR": 40, "BR": 30}
    reg_base = tariff_countries.get(cc, 15)
    # Check news for tariff mentions
    tariff_mentions = sum(
        1 for n in news_signals
        if any(w in n.get("title", "").lower() for w in ["tariff", "sanction", "ban", "duty"])
    )
    # Buyer-aware trade agreement check:
    # If buyer has compliance frameworks, check if this supplier's country
    # is covered by those agreements.  Non-member = higher tariff risk.
    compliance_penalty = 0
    buyer_frameworks = bp.get("compliance_frameworks", [])
    if buyer_frameworks:
        from components.ui_helpers import TRADE_AGREEMENT_MEMBERS
        for framework in buyer_frameworks:
            member_countries = TRADE_AGREEMENT_MEMBERS.get(framework, [])
            if member_countries and cc not in member_countries:
                # Supplier is outside this trade agreement — adds risk
                compliance_penalty += 8
        compliance_penalty = min(compliance_penalty, 30)  # cap

    breakdown["Regulatory / Tariff Risk"] = np.clip(
        reg_base + tariff_mentions * 8 + compliance_penalty, 0, 100
    )

    # 9. Certification Compliance  (new buyer-aware factor) ───────────────────
    #    Checks if the supplier's country/profile can meet buyer's cert reqs.
    #    This is scored as a risk factor: higher = more likely to have gaps.
    required_certs = bp.get("required_certifications", [])
    if required_certs:
        cert_names_required = {c.split("(")[0].strip() for c in required_certs}
        # Map common cert names to the certifications in supplier data
        cert_risk = 0
        for req in cert_names_required:
            # Countries with strong automotive cert ecosystems
            strong_cert_countries = {
                "IATF 16949": ["US", "DE", "JP", "KR", "CA", "FR", "IT", "GB", "CZ", "SK", "MX"],
                "ISO 9001":   list(COUNTRY_GEOPOLITICAL_RISK.keys()),  # widely available
                "ISO 14001":  ["US", "DE", "JP", "KR", "CA", "FR", "IT", "GB", "AU", "CN"],
                "ISO 50001":  ["DE", "JP", "KR", "GB", "FR", "IT"],
                "ISO 45001":  ["US", "DE", "JP", "KR", "CA", "GB", "AU"],
            }
            countries_with_cert = strong_cert_countries.get(req, [])
            if countries_with_cert and cc not in countries_with_cert:
                cert_risk += 15
        breakdown["Certification Compliance"] = np.clip(cert_risk, 0, 100)
    else:
        breakdown["Certification Compliance"] = 0

    # ── Weighted overall score ───────────────────────────────────────────────
    # Add the new factor to weights
    adjusted_weights = dict(RISK_WEIGHTS)
    if required_certs:
        # Re-distribute: take a little from each existing factor
        adjusted_weights["Certification Compliance"] = 0.08
        # Normalize all others down proportionally
        others_total = sum(v for k, v in adjusted_weights.items() if k != "Certification Compliance")
        scale = (1.0 - 0.08) / others_total
        for k in adjusted_weights:
            if k != "Certification Compliance":
                adjusted_weights[k] *= scale

    # Boost weights for user-selected concerns
    concern_mapping = {
        "Geopolitical risks":     "Geopolitical Stability",
        "Shipping / logistics":   "Logistics Reliability",
        "Price volatility":       "Commodity Price Volatility",
        "Single-source dependency": "Supply Concentration",
        "Currency fluctuations":  "Currency Exposure",
        "Climate / natural disasters": "Environmental / Climate",
        "Tariffs / regulations":  "Regulatory / Tariff Risk",
        "Financial instability":  "Financial Health",
    }
    for concern in risk_concerns:
        factor = concern_mapping.get(concern)
        if factor and factor in adjusted_weights:
            adjusted_weights[factor] *= 1.4  # 40% boost

    # Re-normalize
    total_w = sum(adjusted_weights.values())
    adjusted_weights = {k: v / total_w for k, v in adjusted_weights.items()}

    overall = sum(
        breakdown[factor] * adjusted_weights.get(factor, 0)
        for factor in breakdown
    )

    return round(np.clip(overall, 0, 100), 1), breakdown


# ══════════════════════════════════════════════════════════════════════════════
#  DELAY PREDICTION (ML + Rules)
# ══════════════════════════════════════════════════════════════════════════════

def _build_delay_model() -> Tuple[GradientBoostingRegressor, StandardScaler]:
    """
    Train a Gradient Boosting model on synthetic supply-chain data.
    Features: risk_score, commodity_trend, news_sentiment, logistics_score,
              lead_time_days, region_risk
    Target: delay probability (0–100)
    """
    np.random.seed(42)
    n = 500
    risk_scores = np.random.uniform(5, 95, n)
    commodity_trends = np.random.choice([-1, 0, 1], n, p=[0.3, 0.4, 0.3])
    news_sentiments = np.random.uniform(-1, 1, n)
    logistics_scores = np.random.uniform(10, 80, n)
    lead_times = np.random.uniform(7, 90, n)
    region_risks = np.random.uniform(10, 70, n)

    # Synthetic target with realistic correlations
    delay_prob = (
        0.35 * risk_scores
        + 10 * (commodity_trends == 1).astype(float)
        - 8 * news_sentiments
        + 0.25 * logistics_scores
        + 0.15 * lead_times
        + 0.10 * region_risks
        + np.random.normal(0, 5, n)
    )
    delay_prob = np.clip(delay_prob, 0, 100)

    X = np.column_stack([
        risk_scores, commodity_trends, news_sentiments,
        logistics_scores, lead_times, region_risks,
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, random_state=42, learning_rate=0.1,
    )
    model.fit(X_scaled, delay_prob)
    return model, scaler


# Cache the model so it's only trained once per session
_delay_model: Optional[Tuple] = None

def _get_delay_model():
    global _delay_model
    if _delay_model is None:
        _delay_model = _build_delay_model()
    return _delay_model


def predict_delay_probability(
    supplier: Dict,
    risk_score: float,
    commodity_data: Dict,
    news_signals: List[Dict],
) -> Tuple[float, List[Dict]]:
    """
    Predict the probability (0–100%) of a delivery delay in the next 90 days.
    Returns (probability, [detail_factors])
    """
    model, scaler = _get_delay_model()

    # Feature extraction
    steel_data = commodity_data.get("PPI — Steel Mill Products", {})
    trend_map = {"rising": 1, "stable": 0, "falling": -1}
    commodity_trend = trend_map.get(steel_data.get("trend", "stable"), 0)

    # Aggregate news sentiment
    sentiments = []
    for n in news_signals:
        s = n.get("sentiment", "neutral")
        sentiments.append(-1 if s == "negative" else 1 if s == "positive" else 0)
    avg_sentiment = np.mean(sentiments) if sentiments else 0

    cc = supplier.get("country_code", "US")
    region = _get_region(cc)
    logistics_score = REGION_LOGISTICS_RISK.get(region, 30)
    lead_time = supplier.get("lead_time_days", 30)
    region_risk = COUNTRY_GEOPOLITICAL_RISK.get(cc, 30)

    features = np.array([[
        risk_score, commodity_trend, avg_sentiment,
        logistics_score, lead_time, region_risk,
    ]])
    features_scaled = scaler.transform(features)
    prediction = float(model.predict(features_scaled)[0])
    delay_prob = np.clip(prediction, 0, 100)

    # Build explanatory factors
    details = []
    if commodity_trend == 1:
        details.append({
            "factor": "Rising Steel Prices",
            "description": f"Steel PPI has increased {steel_data.get('pct_change_6m', 0):.1f}% "
                           f"over the past 6 months, indicating tightening supply.",
            "severity": "high" if steel_data.get("pct_change_6m", 0) > 5 else "medium",
        })

    neg_news = sum(1 for n in news_signals if n.get("sentiment") == "negative")
    if neg_news >= 3:
        details.append({
            "factor": "Negative News Signals",
            "description": f"{neg_news} negative news articles detected related to "
                           f"steel supply chain or this supplier's region.",
            "severity": "high" if neg_news >= 5 else "medium",
        })

    if logistics_score >= 40:
        details.append({
            "factor": "Regional Logistics Challenges",
            "description": f"The {region} region has above-average logistics risk "
                           f"(score {logistics_score}/100).",
            "severity": "high" if logistics_score >= 50 else "medium",
        })

    if lead_time > 45:
        details.append({
            "factor": "Extended Lead Time",
            "description": f"Supplier lead time of {lead_time} days increases "
                           f"exposure to disruption events.",
            "severity": "medium",
        })

    if region_risk >= 45:
        details.append({
            "factor": "Geopolitical Instability",
            "description": f"Country risk score of {region_risk}/100 "
                           f"indicates elevated geopolitical concerns.",
            "severity": "high" if region_risk >= 60 else "medium",
        })

    return round(delay_prob, 1), details


def generate_risk_breakdown(results: List[Dict]) -> pd.DataFrame:
    """Create a summary DataFrame of all suppliers' risk breakdowns."""
    rows = []
    for r in results:
        row = {"Supplier": r["supplier"]["name"], "Overall Risk": r["risk_score"]}
        row.update(r["breakdown"])
        row["Delay Prob (%)"] = r["delay_probability"]
        rows.append(row)
    return pd.DataFrame(rows)

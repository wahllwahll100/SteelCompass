"""
utils/vendor_recommender.py
===========================
Alternative vendor recommendation engine.

Maintains a curated database of global steel suppliers suitable for
Tier 2 automotive parts manufacturing.  Vendors are ranked by a
composite score of reliability, proximity, cost, and compatibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# ══════════════════════════════════════════════════════════════════════════════
#  VENDOR DATABASE — Global steel suppliers for automotive parts
#  Real companies with publicly available information.
# ══════════════════════════════════════════════════════════════════════════════

VENDOR_DATABASE: List[Dict[str, Any]] = [
    # ── North America ────────────────────────────────────────────────────────
    {
        "name": "Nucor Corporation",
        "country": "US", "country_code": "US", "region": "North America",
        "city": "Charlotte, NC",
        "specialties": ["hot-rolled coils", "cold-rolled", "galvanized", "AHSS", "plate"],
        "materials": ["carbon steel", "high-strength steel", "alloy steel"],
        "annual_capacity_mt": 27_000_000,
        "reliability_score": 92, "years_in_business": 58,
        "certifications": ["IATF 16949", "ISO 9001", "ISO 14001"],
        "lead_time_days": 14, "min_order_mt": 20,
        "cost_index": 105,  # 100 = global avg
        "sustainability_rating": "A",
    },
    {
        "name": "Steel Dynamics Inc (SDI)",
        "country": "US", "country_code": "US", "region": "North America",
        "city": "Fort Wayne, IN",
        "specialties": ["hot-rolled coils", "cold-rolled", "galvanized", "painted"],
        "materials": ["carbon steel", "high-strength steel"],
        "annual_capacity_mt": 13_000_000,
        "reliability_score": 89, "years_in_business": 31,
        "certifications": ["IATF 16949", "ISO 9001"],
        "lead_time_days": 12, "min_order_mt": 25,
        "cost_index": 102,
        "sustainability_rating": "A-",
    },
    {
        "name": "Stelco Holdings",
        "country": "Canada", "country_code": "CA", "region": "North America",
        "city": "Hamilton, ON",
        "specialties": ["hot-rolled coils", "cold-rolled", "galvanized"],
        "materials": ["carbon steel", "high-strength steel"],
        "annual_capacity_mt": 2_800_000,
        "reliability_score": 84, "years_in_business": 113,
        "certifications": ["ISO 9001", "ISO 14001"],
        "lead_time_days": 16, "min_order_mt": 30,
        "cost_index": 100,
        "sustainability_rating": "B+",
    },
    {
        "name": "Ternium México",
        "country": "Mexico", "country_code": "MX", "region": "North America",
        "city": "Monterrey, NL",
        "specialties": ["hot-rolled coils", "cold-rolled", "galvanized", "pre-painted"],
        "materials": ["carbon steel", "high-strength steel"],
        "annual_capacity_mt": 6_500_000,
        "reliability_score": 82, "years_in_business": 24,
        "certifications": ["IATF 16949", "ISO 9001"],
        "lead_time_days": 18, "min_order_mt": 40,
        "cost_index": 92,
        "sustainability_rating": "B",
    },
    # ── Europe ───────────────────────────────────────────────────────────────
    {
        "name": "thyssenkrupp Steel Europe",
        "country": "Germany", "country_code": "DE", "region": "Western Europe",
        "city": "Duisburg",
        "specialties": ["AHSS", "UHSS", "hot-dip galvanized", "electro-galvanized", "tailored blanks"],
        "materials": ["high-strength steel", "ultra-high-strength steel", "alloy steel"],
        "annual_capacity_mt": 11_000_000,
        "reliability_score": 93, "years_in_business": 210,
        "certifications": ["IATF 16949", "ISO 9001", "ISO 14001", "ISO 50001"],
        "lead_time_days": 21, "min_order_mt": 25,
        "cost_index": 118,
        "sustainability_rating": "A",
    },
    {
        "name": "ArcelorMittal Europe",
        "country": "Luxembourg", "country_code": "DE", "region": "Western Europe",
        "city": "Multiple EU sites",
        "specialties": ["AHSS", "press-hardened", "galvanized", "coated", "electrical"],
        "materials": ["carbon steel", "high-strength steel", "ultra-high-strength steel"],
        "annual_capacity_mt": 35_000_000,
        "reliability_score": 90, "years_in_business": 18,
        "certifications": ["IATF 16949", "ISO 9001", "ISO 14001"],
        "lead_time_days": 24, "min_order_mt": 50,
        "cost_index": 115,
        "sustainability_rating": "A-",
    },
    {
        "name": "Voestalpine Steel Division",
        "country": "Austria", "country_code": "DE", "region": "Western Europe",
        "city": "Linz",
        "specialties": ["AHSS", "galvanized", "electro-galvanized", "phs-ultraform"],
        "materials": ["high-strength steel", "ultra-high-strength steel"],
        "annual_capacity_mt": 5_000_000,
        "reliability_score": 94, "years_in_business": 70,
        "certifications": ["IATF 16949", "ISO 9001", "ISO 14001"],
        "lead_time_days": 22, "min_order_mt": 20,
        "cost_index": 122,
        "sustainability_rating": "A+",
    },
    # ── Asia ─────────────────────────────────────────────────────────────────
    {
        "name": "POSCO",
        "country": "South Korea", "country_code": "KR", "region": "East Asia",
        "city": "Pohang",
        "specialties": ["AHSS", "hot-rolled", "cold-rolled", "galvanized", "GIGA STEEL"],
        "materials": ["carbon steel", "high-strength steel", "ultra-high-strength steel", "stainless"],
        "annual_capacity_mt": 42_000_000,
        "reliability_score": 91, "years_in_business": 56,
        "certifications": ["IATF 16949", "ISO 9001", "ISO 14001"],
        "lead_time_days": 35, "min_order_mt": 100,
        "cost_index": 88,
        "sustainability_rating": "A",
    },
    {
        "name": "Nippon Steel Corporation",
        "country": "Japan", "country_code": "JP", "region": "East Asia",
        "city": "Tokyo",
        "specialties": ["AHSS", "ultra-high-tensile", "hot-dip galvanized", "tin-plate"],
        "materials": ["carbon steel", "high-strength steel", "ultra-high-strength steel"],
        "annual_capacity_mt": 44_000_000,
        "reliability_score": 95, "years_in_business": 54,
        "certifications": ["IATF 16949", "ISO 9001", "ISO 14001"],
        "lead_time_days": 38, "min_order_mt": 80,
        "cost_index": 95,
        "sustainability_rating": "A",
    },
    {
        "name": "JFE Steel Corporation",
        "country": "Japan", "country_code": "JP", "region": "East Asia",
        "city": "Tokyo",
        "specialties": ["hot-rolled", "cold-rolled", "galvanized", "high-tensile"],
        "materials": ["carbon steel", "high-strength steel", "alloy steel"],
        "annual_capacity_mt": 26_000_000,
        "reliability_score": 90, "years_in_business": 21,
        "certifications": ["IATF 16949", "ISO 9001"],
        "lead_time_days": 36, "min_order_mt": 60,
        "cost_index": 93,
        "sustainability_rating": "A-",
    },
    {
        "name": "Baosteel (Baoshan Iron & Steel)",
        "country": "China", "country_code": "CN", "region": "East Asia",
        "city": "Shanghai",
        "specialties": ["hot-rolled", "cold-rolled", "galvanized", "automotive plate"],
        "materials": ["carbon steel", "high-strength steel", "stainless"],
        "annual_capacity_mt": 50_000_000,
        "reliability_score": 80, "years_in_business": 46,
        "certifications": ["IATF 16949", "ISO 9001"],
        "lead_time_days": 40, "min_order_mt": 200,
        "cost_index": 78,
        "sustainability_rating": "B",
    },
    {
        "name": "Hyundai Steel",
        "country": "South Korea", "country_code": "KR", "region": "East Asia",
        "city": "Incheon",
        "specialties": ["hot-rolled", "cold-rolled", "galvanized", "electrical"],
        "materials": ["carbon steel", "high-strength steel"],
        "annual_capacity_mt": 21_000_000,
        "reliability_score": 87, "years_in_business": 71,
        "certifications": ["IATF 16949", "ISO 9001", "ISO 14001"],
        "lead_time_days": 33, "min_order_mt": 80,
        "cost_index": 86,
        "sustainability_rating": "B+",
    },
    # ── India ────────────────────────────────────────────────────────────────
    {
        "name": "Tata Steel",
        "country": "India", "country_code": "IN", "region": "South Asia",
        "city": "Jamshedpur",
        "specialties": ["hot-rolled", "cold-rolled", "galvanized", "Tata Steelium"],
        "materials": ["carbon steel", "high-strength steel", "alloy steel"],
        "annual_capacity_mt": 34_000_000,
        "reliability_score": 83, "years_in_business": 117,
        "certifications": ["IATF 16949", "ISO 9001", "ISO 14001"],
        "lead_time_days": 42, "min_order_mt": 100,
        "cost_index": 75,
        "sustainability_rating": "B+",
    },
    {
        "name": "JSW Steel",
        "country": "India", "country_code": "IN", "region": "South Asia",
        "city": "Mumbai",
        "specialties": ["hot-rolled", "cold-rolled", "galvanized", "color-coated"],
        "materials": ["carbon steel", "high-strength steel"],
        "annual_capacity_mt": 28_000_000,
        "reliability_score": 79, "years_in_business": 42,
        "certifications": ["ISO 9001", "ISO 14001"],
        "lead_time_days": 45, "min_order_mt": 120,
        "cost_index": 73,
        "sustainability_rating": "B",
    },
    # ── Turkey / Brazil ──────────────────────────────────────────────────────
    {
        "name": "Erdemir (Ereğli Iron & Steel)",
        "country": "Turkey", "country_code": "TR", "region": "Middle East",
        "city": "Ereğli",
        "specialties": ["hot-rolled", "cold-rolled", "galvanized", "tin-plate"],
        "materials": ["carbon steel", "high-strength steel"],
        "annual_capacity_mt": 8_500_000,
        "reliability_score": 78, "years_in_business": 59,
        "certifications": ["IATF 16949", "ISO 9001"],
        "lead_time_days": 28, "min_order_mt": 50,
        "cost_index": 82,
        "sustainability_rating": "B",
    },
    {
        "name": "Gerdau S.A.",
        "country": "Brazil", "country_code": "BR", "region": "South America",
        "city": "Porto Alegre",
        "specialties": ["hot-rolled", "special bar quality", "flat steel"],
        "materials": ["carbon steel", "alloy steel"],
        "annual_capacity_mt": 12_000_000,
        "reliability_score": 81, "years_in_business": 123,
        "certifications": ["ISO 9001", "ISO 14001"],
        "lead_time_days": 32, "min_order_mt": 60,
        "cost_index": 80,
        "sustainability_rating": "B+",
    },
]

# ── Material compatibility map ───────────────────────────────────────────────
MATERIAL_KEYWORDS = {
    "high-strength steel coils": ["high-strength steel", "AHSS", "hot-rolled"],
    "cold-rolled steel sheets":  ["cold-rolled", "carbon steel"],
    "galvanized steel":          ["galvanized", "hot-dip galvanized"],
    "AHSS (Advanced High-Strength)": ["AHSS", "high-strength steel", "ultra-high-strength steel"],
    "hot-rolled steel coils":    ["hot-rolled", "carbon steel"],
    "press-hardened steel":      ["press-hardened", "ultra-high-strength steel", "UHSS"],
    "stainless steel":           ["stainless"],
    "electrical steel":          ["electrical"],
    "alloy steel":               ["alloy steel"],
}


def _material_compatibility(vendor: Dict, material_type: str) -> float:
    """Score 0–100 for how well a vendor matches the required material."""
    keywords = MATERIAL_KEYWORDS.get(material_type, [material_type.lower()])
    matches = 0
    total = len(keywords) if keywords else 1

    for kw in keywords:
        kw_lower = kw.lower()
        if any(kw_lower in s.lower() for s in vendor.get("specialties", [])):
            matches += 1
        if any(kw_lower in m.lower() for m in vendor.get("materials", [])):
            matches += 1

    # Also check certifications (IATF 16949 is critical for automotive)
    has_iatf = "IATF 16949" in vendor.get("certifications", [])
    cert_bonus = 15 if has_iatf else 0

    raw = (matches / (total * 2)) * 85 + cert_bonus
    return min(100, raw)


def _proximity_score(
    vendor: Dict,
    current_suppliers: List[Dict],
    buyer_profile: Optional[Dict] = None,
) -> float:
    """
    Score based on how close the vendor is to the BUYER's manufacturing
    sites (primary anchor) and existing supply-chain nodes (secondary).
    Same region as buyer plant = 95, same as existing supplier = 85,
    neighboring = 65, distant = 35.
    """
    vendor_region = vendor.get("region", "")

    # Primary anchor: buyer's manufacturing regions
    buyer_regions = []
    if buyer_profile:
        buyer_regions = buyer_profile.get("manufacturing_regions", [])
        if not buyer_regions:
            hq_cc = buyer_profile.get("hq_country_code", "")
            if hq_cc:
                from components.ui_helpers import REGION_MAP
                buyer_regions = [REGION_MAP.get(hq_cc, "North America")]

    # Secondary anchor: current supplier regions
    supplier_regions = [s.get("region", "North America") for s in current_suppliers]

    # Neighboring regions lookup
    neighbors = {
        "North America": ["South America", "East Asia"],
        "Western Europe": ["Eastern Europe", "Middle East", "North America"],
        "Eastern Europe": ["Western Europe", "Middle East"],
        "East Asia": ["Southeast Asia", "South Asia", "North America"],
        "South Asia": ["East Asia", "Southeast Asia", "Middle East"],
        "Southeast Asia": ["East Asia", "South Asia", "Oceania"],
        "Middle East": ["South Asia", "Eastern Europe", "Western Europe", "Africa"],
        "South America": ["North America"],
        "Oceania": ["East Asia", "Southeast Asia"],
        "Africa": ["Middle East", "Western Europe"],
    }

    # Score from buyer proximity (weighted heavier)
    buyer_score = 0
    if buyer_regions:
        if vendor_region in buyer_regions:
            buyer_score = 95
        elif any(vendor_region in neighbors.get(r, []) for r in buyer_regions):
            buyer_score = 65
        else:
            buyer_score = 30

    # Score from supplier proximity
    supplier_score = 0
    if vendor_region in supplier_regions:
        supplier_score = 85
    elif any(vendor_region in neighbors.get(r, []) for r in supplier_regions):
        supplier_score = 60
    else:
        supplier_score = 35

    # Blend: 70% buyer proximity, 30% existing supplier proximity
    if buyer_regions:
        return round(buyer_score * 0.7 + supplier_score * 0.3, 1)
    else:
        return supplier_score


def _certification_compliance_score(
    vendor: Dict,
    buyer_profile: Optional[Dict] = None,
) -> float:
    """
    Score 0–100 for how well a vendor meets the buyer's certification
    requirements.  100 = fully compliant (all certs met), 0 = no match.
    Returns 50 (neutral) if buyer has no requirements specified.
    """
    if not buyer_profile:
        return 50

    required = buyer_profile.get("required_certifications", [])
    if not required:
        return 50

    vendor_certs = set(vendor.get("certifications", []))
    matches = 0
    for req in required:
        # Match on the short name (before the parenthetical)
        req_short = req.split("(")[0].strip()
        if any(req_short in vc for vc in vendor_certs):
            matches += 1

    return round((matches / len(required)) * 100, 1)


def _trade_agreement_score(
    vendor: Dict,
    buyer_profile: Optional[Dict] = None,
) -> float:
    """
    Score 0–100 for trade agreement alignment between vendor country
    and buyer's compliance frameworks.  Higher = better aligned.
    Returns 50 (neutral) if buyer has no frameworks selected.
    """
    if not buyer_profile:
        return 50

    frameworks = buyer_profile.get("compliance_frameworks", [])
    if not frameworks:
        return 50

    from components.ui_helpers import TRADE_AGREEMENT_MEMBERS

    vendor_cc = vendor.get("country_code", "")
    aligned = 0
    relevant = 0

    for fw in frameworks:
        members = TRADE_AGREEMENT_MEMBERS.get(fw, [])
        if members:  # only count frameworks that have a member list
            relevant += 1
            if vendor_cc in members:
                aligned += 1

    if relevant == 0:
        return 50

    return round((aligned / relevant) * 100, 1)


def _cost_score(vendor: Dict) -> float:
    """Invert cost_index to a 0–100 score (lower cost = higher score)."""
    idx = vendor.get("cost_index", 100)
    return max(0, min(100, 150 - idx))


def get_alternative_vendors(
    material_type: str,
    current_suppliers: List[Dict],
    risk_results: List[Dict],
    commodity_data: Dict,
    buyer_profile: Optional[Dict] = None,
) -> List[Dict]:
    """
    Filter and enrich alternative vendors from the database.
    Excludes current suppliers by name.
    When buyer_profile is provided, vendors are also scored on:
      - Proximity to buyer's manufacturing sites (not just existing suppliers)
      - Certification compliance with buyer requirements
      - Trade agreement alignment with buyer's compliance frameworks
    """
    current_names = {s["name"].lower() for s in current_suppliers}
    alternatives = []

    for vendor in VENDOR_DATABASE:
        if vendor["name"].lower() in current_names:
            continue

        compat = _material_compatibility(vendor, material_type)
        if compat < 25:  # too low compatibility, skip
            continue

        proximity = _proximity_score(vendor, current_suppliers, buyer_profile)
        cost = _cost_score(vendor)
        cert_score = _certification_compliance_score(vendor, buyer_profile)
        trade_score = _trade_agreement_score(vendor, buyer_profile)

        alternatives.append({
            **vendor,
            "compatibility_score": round(compat, 1),
            "proximity_score": round(proximity, 1),
            "cost_score": round(cost, 1),
            "cert_compliance_score": cert_score,
            "trade_agreement_score": trade_score,
        })

    return alternatives


def rank_vendors(
    alternatives: List[Dict],
    material_type: str,
    weights: Optional[Dict] = None,
    buyer_profile: Optional[Dict] = None,
) -> List[Dict]:
    """
    Rank alternative vendors by composite score.
    When buyer_profile is present, adds certification compliance
    and trade agreement alignment into the composite weighting.

    Default weights (without buyer):
        reliability 30%, compatibility 30%, cost 20%, proximity 20%
    With buyer profile:
        reliability 22%, compatibility 22%, cost 14%, proximity 16%,
        cert compliance 14%, trade alignment 12%
    """
    has_buyer = buyer_profile is not None and bool(buyer_profile.get("company_name"))

    if weights is None:
        if has_buyer:
            weights = {
                "reliability": 0.22,
                "compatibility": 0.22,
                "cost": 0.14,
                "proximity": 0.16,
                "cert_compliance": 0.14,
                "trade_alignment": 0.12,
            }
        else:
            weights = {
                "reliability": 0.30,
                "compatibility": 0.30,
                "cost": 0.20,
                "proximity": 0.20,
                "cert_compliance": 0.0,
                "trade_alignment": 0.0,
            }

    for v in alternatives:
        composite = (
            v.get("reliability_score", 50) * weights.get("reliability", 0)
            + v.get("compatibility_score", 50) * weights.get("compatibility", 0)
            + v.get("cost_score", 50) * weights.get("cost", 0)
            + v.get("proximity_score", 50) * weights.get("proximity", 0)
            + v.get("cert_compliance_score", 50) * weights.get("cert_compliance", 0)
            + v.get("trade_agreement_score", 50) * weights.get("trade_alignment", 0)
        )
        v["composite_score"] = round(composite, 1)

    ranked = sorted(alternatives, key=lambda x: x["composite_score"], reverse=True)

    # Add rank
    for i, v in enumerate(ranked, 1):
        v["rank"] = i

    return ranked

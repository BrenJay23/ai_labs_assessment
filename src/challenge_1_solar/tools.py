from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from rapidfuzz import process

from .model import (
    VALID_CITIES,
    _city_pvout,
    _holdout,
    _feature_cols,
    predict_pvout,
    compute_yield,
    GCR,
    PANEL_EFFICIENCY,
)
from .schemas import WeatherInput


def _resolve_city(city: str) -> tuple[str, float] | tuple[None, None]:
    match, score, _ = process.extractOne(city, VALID_CITIES)
    return (match, score) if score >= 80 else (None, None)


@tool
def predict_solar_yield(
    city: str,
    farm_area_ha: float,
    date: Optional[str] = None,
    weather: Optional[WeatherInput] = None,
    gcr: float = GCR,
    panel_efficiency: float = PANEL_EFFICIENCY,
) -> dict:
    """
    Predict daily solar energy yield for an Australian solar farm.

    Uses exact 2010 historical weather observation for the given city and date
    (month and day matched, year mapped to 2010). If no date is provided, uses
    the city's annual average as the baseline. Any provided weather conditions
    override the historical baseline — missing fields fall back to historical average.

    Yield Formula:
        Panel area (m²)    = farm_area_ha × 10,000 × GCR
        Installed kWp      = Panel area × panel_efficiency
        Daily yield (kWh)  = Installed kWp × PVOUT (kWh/kWp/day)

    Args:
        city            : Australian city name (e.g. 'Sydney', 'Alice Springs')
        farm_area_ha    : Farm area in hectares. Convert first if needed:
                          1 acre=0.4047ha, 1 km²=100ha, 1 m²=0.0001ha
        date            : Date in YYYY-MM-DD format. Resolve relative dates like
                          'tomorrow' or 'next Monday' using today's date before calling.
        weather         : Optional weather conditions. Infer as many fields as possible
                          from the user's description. Leave unknown fields as None —
                          they will be filled with historical averages for the city.
        gcr             : Ground Coverage Ratio (default 0.35). Typical range: 0.2–0.5.
                          Higher GCR means more panels per unit area.
        panel_efficiency: Solar panel efficiency as decimal (default 0.18 = 18%).
                          Typical commercial range: 0.15–0.25.
    """
    resolved_city, score = _resolve_city(city)
    if resolved_city is None:
        return {"error": f"City '{city}' not found.", "available_cities": VALID_CITIES}

    weather_dict = weather.to_model_dict() if weather else None
    pvout = predict_pvout(resolved_city, date, weather_dict)
    result = compute_yield(pvout, farm_area_ha, gcr, panel_efficiency)

    assumptions = []
    if resolved_city.lower().replace(" ", "") != city.lower().replace(" ", ""):
        assumptions.append(
            f"Interpreted '{city}' as '{resolved_city}' (match: {score:.0f}%)"
        )
    if date:
        ref_date = pd.Timestamp(date).replace(year=2010)
        assumptions.append(
            f"Mapped date to 2010-{ref_date.month:02d}-{ref_date.day:02d} historical weather for {resolved_city}"
        )
    else:
        assumptions.append(
            f"No date provided — used annual historical average for {resolved_city}"
        )
    if weather_dict:
        provided = [k for k, v in weather_dict.items() if v is not None]
        missing = [k for k, v in weather_dict.items() if v is None]
        if provided:
            assumptions.append(f"Weather overrides applied: {provided}")
        if missing:
            assumptions.append(
                f"Missing weather fields filled with historical average: {missing}"
            )

    assumptions.append(f"GCR={gcr}, panel efficiency={panel_efficiency * 100:.0f}%")

    return {
        "city": resolved_city,
        "farm_area_ha": farm_area_ha,
        "date": date,
        **result,
        "assumptions": assumptions,
    }


@tool
def get_city_solar_stats(city: Optional[str] = None) -> dict:
    """
    Get reference solar and weather statistics for an Australian city.
    Call this first if unsure about the exact city name or to discover available cities.
    Also useful for comparing cities without needing a farm size.

    Args:
        city: Australian city name. If None or not found, returns list of available cities.
    """
    if city is None:
        return {"available_cities": VALID_CITIES}

    resolved_city, score = _resolve_city(city)
    if resolved_city is None:
        return {"error": f"City '{city}' not found.", "available_cities": VALID_CITIES}

    city_df = _holdout[_holdout["Location"] == resolved_city]
    avg = city_df[_feature_cols].mean(numeric_only=True).round(2)

    return {
        "city": resolved_city,
        "gsa_yearly_pvout": _city_pvout.get(resolved_city),
        "avg_humidity_3pm": avg.get("Humidity3pm"),
        "avg_temp_9am": avg.get("Temp9am"),
        "avg_sunshine_hrs": avg.get("Sunshine"),
        "avg_cloud_9am": avg.get("Cloud9am"),
        "avg_cloud_3pm": avg.get("Cloud3pm"),
        "avg_max_temp": avg.get("MaxTemp"),
        "avg_rainfall_mm": avg.get("Rainfall"),
        "data_year": 2010,
        "note": "Weather averages from 2010 historical data. PVOUT in kWh/kWp/day from Global Solar Atlas.",
    }

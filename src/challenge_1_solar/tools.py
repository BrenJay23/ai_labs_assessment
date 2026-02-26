from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from rapidfuzz import process

from .model import (
    VALID_CITIES,
    HOLDOUT,
    predict_pvout,
    compute_yield,
    GCR,
    PANEL_EFFICIENCY,
    get_weather_from_row,
    get_weather_from_avg,
)
from .schemas import WeatherInput


def _resolve_city(city: str) -> tuple[str, float] | tuple[None, None]:
    match, score, _ = process.extractOne(city, VALID_CITIES)
    return (match, score) if score >= 80 else (None, None)


@tool
def get_city_weather_stats(
    city: str,
    month: Optional[int] = None,
    date: Optional[str] = None,
) -> dict:
    """
    Get weather conditions for an Australian city. Used to fetch a baseline
    before calling predict_solar_yield, or to answer weather-related questions.

    Priority: date > month > yearly average.

    Args:
        city  : Australian city name.
        month : Month as integer (1=Jan, 12=Dec). Returns average for that month.
        date  : Specific date in YYYY-MM-DD format. Returns the exact historical
                observation for that date (mapped to 2010).
    """
    resolved_city, score = _resolve_city(city)
    if resolved_city is None:
        return {"error": f"City '{city}' not found.", "available_cities": VALID_CITIES}

    city_df = HOLDOUT[HOLDOUT["Location"] == resolved_city]
    period_label = "yearly average"

    if date:
        ref_date = pd.Timestamp(date).replace(year=2010)
        row = city_df[city_df["Date"] == ref_date]
        if not row.empty:
            weather = get_weather_from_row(row.iloc[0])
            period_label = f"{ref_date.strftime('%B %d')} (2010 historical)"
        else:
            # fallback to monthly average if exact date missing
            month = ref_date.month
            month_df = city_df[city_df["Date"].dt.month == month]
            weather = get_weather_from_avg(month_df)
            period_label = f"{ref_date.strftime('%B')} average (date not found in data)"
    elif month:
        month_df = city_df[city_df["Date"].dt.month == month]
        weather = get_weather_from_avg(month_df)
        period_label = f"{pd.Timestamp(2010, month, 1).strftime('%B')} average"
    else:
        weather = get_weather_from_avg(city_df)

    return {
        "city": resolved_city,
        "period": period_label,
        "weather": weather,
    }


@tool
def predict_solar_yield(
    city: str,
    farm_area_ha: float,
    weather: WeatherInput,
    gcr: float = GCR,
    panel_efficiency: float = PANEL_EFFICIENCY,
) -> dict:
    """
    Predict daily solar energy yield for an Australian solar farm.

    Always call get_city_weather_stats first to fetch a baseline weather dict,
    then override specific fields with the user's described conditions before calling this.

    Yield Formula:
        Panel area (m²)    = farm_area_ha × 10,000 × GCR
        Installed kWp      = Panel area × panel_efficiency
        Daily yield (kWh)  = Installed kWp × PVOUT (kWh/kWp/day)

    Args:
        city            : Australian city name.
        farm_area_ha    : Farm area in hectares. Convert first if needed:
                          1 acre=0.4047ha, 1 km²=100ha, 1 m²=0.0001ha
        weather         : Weather conditions from get_city_weather_stats, with any
                          user-described overrides applied on top.
        gcr             : Ground Coverage Ratio (default 0.35). Typical range: 0.2–0.5.
        panel_efficiency: Solar panel efficiency as decimal (default 0.18 = 18%).
    """
    resolved_city, score = _resolve_city(city)
    if resolved_city is None:
        return {"error": f"City '{city}' not found.", "available_cities": VALID_CITIES}

    weather_dict = weather.to_model_dict()
    pvout = predict_pvout(weather_dict)
    result = compute_yield(pvout, farm_area_ha, gcr, panel_efficiency)

    return {
        "city": resolved_city,
        "farm_area_ha": farm_area_ha,
        **result,
    }

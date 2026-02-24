from typing import Optional
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
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

_stats = _holdout[_feature_cols].describe()


def _fdesc(col: str, unit: str) -> str:
    if col not in _stats:
        return unit
    return (
        f"{unit}. "
        f"Typical range: {_stats[col]['25%']:.1f}–{_stats[col]['75%']:.1f}, "
        f"average: {_stats[col]['mean']:.1f}"
    )


class WeatherInput(BaseModel):
    """Optional weather conditions. Leave fields as None if unknown — missing fields
    will be filled with historical averages for the city."""

    min_temp: Optional[float] = Field(
        None, description=_fdesc("MinTemp", "Min temperature (°C)")
    )
    max_temp: Optional[float] = Field(
        None, description=_fdesc("MaxTemp", "Max temperature (°C)")
    )
    rainfall: Optional[float] = Field(
        None, description=_fdesc("Rainfall", "Rainfall (mm)")
    )
    evaporation: Optional[float] = Field(
        None, description=_fdesc("Evaporation", "Evaporation (mm)")
    )
    sunshine: Optional[float] = Field(
        None, description=_fdesc("Sunshine", "Sunshine hours")
    )
    wind_gust_speed: Optional[float] = Field(
        None, description=_fdesc("WindGustSpeed", "Wind gust speed (km/h)")
    )
    wind_speed_9am: Optional[float] = Field(
        None, description=_fdesc("WindSpeed9am", "Wind speed at 9am (km/h)")
    )
    wind_speed_3pm: Optional[float] = Field(
        None, description=_fdesc("WindSpeed3pm", "Wind speed at 3pm (km/h)")
    )
    humidity_9am: Optional[float] = Field(
        None, description=_fdesc("Humidity9am", "Humidity at 9am (%)")
    )
    humidity_3pm: Optional[float] = Field(
        None, description=_fdesc("Humidity3pm", "Humidity at 3pm (%)")
    )
    pressure_9am: Optional[float] = Field(
        None, description=_fdesc("Pressure9am", "Atmospheric pressure at 9am (hPa)")
    )
    pressure_3pm: Optional[float] = Field(
        None, description=_fdesc("Pressure3pm", "Atmospheric pressure at 3pm (hPa)")
    )
    cloud_9am: Optional[float] = Field(
        None,
        description=_fdesc(
            "Cloud9am", "Cloud cover at 9am (eighths, 0–8). Clear=0, Overcast=8"
        ),
    )
    cloud_3pm: Optional[float] = Field(
        None,
        description=_fdesc(
            "Cloud3pm", "Cloud cover at 3pm (eighths, 0–8). Clear=0, Overcast=8"
        ),
    )
    temp_9am: Optional[float] = Field(
        None, description=_fdesc("Temp9am", "Temperature at 9am (°C)")
    )
    temp_3pm: Optional[float] = Field(
        None, description=_fdesc("Temp3pm", "Temperature at 3pm (°C)")
    )
    rain_today: Optional[str] = Field(
        None, description="Whether it is raining today. 'Yes' or 'No'"
    )


def _to_model_dict(w: WeatherInput) -> dict:
    return {
        "MinTemp": w.min_temp,
        "MaxTemp": w.max_temp,
        "Rainfall": w.rainfall,
        "Evaporation": w.evaporation,
        "Sunshine": w.sunshine,
        "WindGustSpeed": w.wind_gust_speed,
        "WindSpeed9am": w.wind_speed_9am,
        "WindSpeed3pm": w.wind_speed_3pm,
        "Humidity9am": w.humidity_9am,
        "Humidity3pm": w.humidity_3pm,
        "Pressure9am": w.pressure_9am,
        "Pressure3pm": w.pressure_3pm,
        "Cloud9am": w.cloud_9am,
        "Cloud3pm": w.cloud_3pm,
        "Temp9am": w.temp_9am,
        "Temp3pm": w.temp_3pm,
        "RainToday": w.rain_today,
    }


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

    weather_dict = _to_model_dict(weather) if weather else None
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

    assumptions += [
        f"GCR={gcr}, panel efficiency={panel_efficiency * 100:.0f}%",
        "Performance ratio already included in GSA PVOUT",
    ]

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
        "avg_humidity_3pm": avg.get("Humidity3pm"),  # #1 feature importance
        "avg_temp_9am": avg.get("Temp9am"),  # #2
        "avg_sunshine_hrs": avg.get("Sunshine"),  # #3 (also most intuitive)
        "avg_cloud_9am": avg.get("Cloud9am"),  # #4
        "avg_cloud_3pm": avg.get("Cloud3pm"),  # #5
        "avg_max_temp": avg.get("MaxTemp"),
        "avg_rainfall_mm": avg.get("Rainfall"),
        "data_year": 2010,
        "note": "Weather averages from 2010 historical data. PVOUT in kWh/kWp/day from Global Solar Atlas.",
    }

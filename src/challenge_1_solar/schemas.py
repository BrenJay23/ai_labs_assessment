from typing import Optional
from pydantic import BaseModel, Field

from .model import HOLDOUT, FEATURE_COLS


def _fdesc(col: str, unit: str) -> str:
    stats = HOLDOUT[FEATURE_COLS].describe()
    if col not in stats:
        return unit
    return (
        f"{unit}. "
        f"Typical range: {stats[col]['25%']:.1f}–{stats[col]['75%']:.1f}, "
        f"average: {stats[col]['mean']:.1f}"
    )


class WeatherInput(BaseModel):
    """Weather conditions for solar yield prediction."""

    model_config = {"populate_by_name": True}

    min_temp: Optional[float] = Field(
        None, alias="MinTemp", description=_fdesc("MinTemp", "Min temperature (°C)")
    )
    max_temp: Optional[float] = Field(
        None, alias="MaxTemp", description=_fdesc("MaxTemp", "Max temperature (°C)")
    )
    rainfall: Optional[float] = Field(
        None, alias="Rainfall", description=_fdesc("Rainfall", "Rainfall (mm)")
    )
    evaporation: Optional[float] = Field(
        None, alias="Evaporation", description=_fdesc("Evaporation", "Evaporation (mm)")
    )
    sunshine: Optional[float] = Field(
        None, alias="Sunshine", description=_fdesc("Sunshine", "Sunshine hours")
    )
    wind_gust_speed: Optional[float] = Field(
        None,
        alias="WindGustSpeed",
        description=_fdesc("WindGustSpeed", "Wind gust speed (km/h)"),
    )
    wind_speed_9am: Optional[float] = Field(
        None,
        alias="WindSpeed9am",
        description=_fdesc("WindSpeed9am", "Wind speed at 9am (km/h)"),
    )
    wind_speed_3pm: Optional[float] = Field(
        None,
        alias="WindSpeed3pm",
        description=_fdesc("WindSpeed3pm", "Wind speed at 3pm (km/h)"),
    )
    humidity_9am: Optional[float] = Field(
        None,
        alias="Humidity9am",
        description=_fdesc("Humidity9am", "Humidity at 9am (%)"),
    )
    humidity_3pm: Optional[float] = Field(
        None,
        alias="Humidity3pm",
        description=_fdesc("Humidity3pm", "Humidity at 3pm (%)"),
    )
    pressure_9am: Optional[float] = Field(
        None,
        alias="Pressure9am",
        description=_fdesc("Pressure9am", "Atmospheric pressure at 9am (hPa)"),
    )
    pressure_3pm: Optional[float] = Field(
        None,
        alias="Pressure3pm",
        description=_fdesc("Pressure3pm", "Atmospheric pressure at 3pm (hPa)"),
    )
    cloud_9am: Optional[float] = Field(
        None,
        alias="Cloud9am",
        description=_fdesc(
            "Cloud9am", "Cloud cover at 9am (eighths, 0–8). Clear=0, Overcast=8"
        ),
    )
    cloud_3pm: Optional[float] = Field(
        None,
        alias="Cloud3pm",
        description=_fdesc(
            "Cloud3pm", "Cloud cover at 3pm (eighths, 0–8). Clear=0, Overcast=8"
        ),
    )
    temp_9am: Optional[float] = Field(
        None, alias="Temp9am", description=_fdesc("Temp9am", "Temperature at 9am (°C)")
    )
    temp_3pm: Optional[float] = Field(
        None, alias="Temp3pm", description=_fdesc("Temp3pm", "Temperature at 3pm (°C)")
    )
    wind_gust_dir: Optional[str] = Field(
        None, alias="WindGustDir", description="Wind gust direction (e.g. 'N', 'SW')"
    )
    wind_dir_9am: Optional[str] = Field(
        None, alias="WindDir9am", description="Wind direction at 9am (e.g. 'N', 'SW')"
    )
    wind_dir_3pm: Optional[str] = Field(
        None, alias="WindDir3pm", description="Wind direction at 3pm (e.g. 'N', 'SW')"
    )
    rain_today: Optional[str] = Field(
        None,
        alias="RainToday",
        description="Whether it is raining today. 'Yes' or 'No'",
    )

    def to_model_dict(self) -> dict:
        return {
            k: v for k, v in self.model_dump(by_alias=True).items() if v is not None
        }

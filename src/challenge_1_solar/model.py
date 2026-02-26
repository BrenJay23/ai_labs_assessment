from pathlib import Path
import joblib
import pandas as pd
import xgboost as xgb

PROCESSED_DIR = Path("data/processed")
CAT_COLS = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
GCR = 0.35
PANEL_EFFICIENCY = 0.18

_MODEL = xgb.XGBRegressor()
_MODEL.load_model(PROCESSED_DIR / "xgb_model.json")
_ENCODERS = joblib.load(PROCESSED_DIR / "encoders.pkl")
FEATURE_COLS = joblib.load(PROCESSED_DIR / "feature_cols.pkl")
HOLDOUT = pd.read_csv(PROCESSED_DIR / "holdout_weather.csv", parse_dates=["Date"])

VALID_CITIES = sorted(HOLDOUT["Location"].unique().tolist())


def get_weather_from_row(row: pd.Series) -> dict:
    """Convert a holdout weather row to a weather dict."""
    return {col: row[col] for col in FEATURE_COLS if col in row.index}


def get_weather_from_avg(city_df: pd.DataFrame) -> dict:
    """Compute average weather conditions for a city DataFrame."""
    avg = city_df[FEATURE_COLS].mean(numeric_only=True)
    for col in CAT_COLS:
        if col in FEATURE_COLS:
            mode = city_df[col].mode()
            avg[col] = mode[0] if not mode.empty else "N"
    return avg.to_dict()


def predict_pvout(weather: dict) -> float:
    """
    Predict PVOUT (kWh/kWp/day) given explicit weather conditions.
    Assumes weather dict is complete — fetch baseline from get_city_weather_stats first.
    """
    baseline = pd.Series(weather)

    for col in CAT_COLS:
        if col in baseline.index and isinstance(baseline[col], str):
            baseline[col] = _ENCODERS[col].transform([str(baseline[col])])[0]

    X = pd.DataFrame([baseline])[FEATURE_COLS]
    return float(_MODEL.predict(X)[0])


def compute_yield(
    pvout: float,
    farm_area_ha: float,
    gcr: float = GCR,
    panel_efficiency: float = PANEL_EFFICIENCY,
) -> dict:
    """
    Yield Formula:
        Panel area (m²)    = farm_area_ha × 10,000 × GCR
        Installed kWp      = Panel area × panel_efficiency
        Daily yield (kWh)  = Installed kWh × PVOUT (kWh/kWp/day)
    """
    panel_area_m2 = farm_area_ha * 10_000 * gcr
    installed_kwp = panel_area_m2 * panel_efficiency
    daily_kwh = installed_kwp * pvout
    daily_mwh = daily_kwh / 1000

    return {
        "installed_kwp": round(installed_kwp, 2),
        "pvout": round(pvout, 4),
        "daily_kwh": round(daily_kwh, 2),
        "daily_mwh": round(daily_mwh, 4),
        "gcr": gcr,
        "panel_efficiency": panel_efficiency,
    }

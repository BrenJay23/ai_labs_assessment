from pathlib import Path
import joblib
import pandas as pd
import xgboost as xgb

PROCESSED_DIR = Path("data/processed")
CAT_COLS = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
GCR = 0.35
PANEL_EFFICIENCY = 0.18

_model = xgb.XGBRegressor()
_model.load_model(PROCESSED_DIR / "xgb_model.json")
_encoders = joblib.load(PROCESSED_DIR / "encoders.pkl")
_feature_cols = joblib.load(PROCESSED_DIR / "feature_cols.pkl")
_city_pvout = joblib.load(PROCESSED_DIR / "city_pvout.pkl")
_holdout = pd.read_csv(PROCESSED_DIR / "holdout_weather.csv", parse_dates=["Date"])

VALID_CITIES = sorted(_holdout["Location"].unique().tolist())


def get_feature_vector(
    city: str, date: str = None, weather: dict = None
) -> pd.DataFrame:
    city_df = _holdout[_holdout["Location"] == city]

    if date:
        ref_date = pd.Timestamp(date).replace(year=2010)
        row = city_df[city_df["Date"] == ref_date]
        if len(row) > 0:
            baseline = row[_feature_cols].iloc[0].copy()
        else:
            # fallback: numeric mean + categorical mode
            baseline = city_df[_feature_cols].select_dtypes(include="number").mean()
            for col in CAT_COLS:
                if col in _feature_cols:
                    baseline[col] = city_df[col].mode()[0]
    else:
        baseline = city_df[_feature_cols].select_dtypes(include="number").mean()
        for col in CAT_COLS:
            if col in _feature_cols:
                baseline[col] = city_df[col].mode()[0]

    if weather:
        for field, value in weather.items():
            if value is not None and field in baseline.index:
                baseline[field] = value

    for col in CAT_COLS:
        if col in baseline.index and isinstance(baseline[col], str):
            baseline[col] = _encoders[col].transform([str(baseline[col])])[0]

    return pd.DataFrame([baseline])[_feature_cols]


def predict_pvout(city: str, date: str = None, weather: dict = None) -> float:
    X = get_feature_vector(city, date, weather)
    return float(_model.predict(X)[0])


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
        Daily yield (kWh)  = Installed kWp × PVOUT (kWh/kWp/day)
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

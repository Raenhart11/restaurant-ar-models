"""Feature engineering shared by training and prediction.

The public dataset contains item-level quantities for many restaurants. For a
single-restaurant prototype, the target is the average total menu-item units
sold per active restaurant per day, not customer order count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

TARGET_COLUMN = "daily_demand"
DATE_COLUMN = "date"
MIN_HISTORY_DAYS = 14

NUMERIC_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "lag1",
    "lag7",
    "lag14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "trend_7",
    "avg_selling_price",
    "avg_market_price",
    "avg_ingredient_cost",
    "promotion_rate",
    "special_event",
]
CATEGORICAL_FEATURES = ["weather_condition"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_RAW_COLUMNS = {
    "date",
    "restaurant_id",
    "typical_ingredient_cost",
    "observed_market_price",
    "actual_selling_price",
    "quantity_sold",
    "has_promotion",
    "special_event",
    "weather_condition",
}


@dataclass(frozen=True)
class ForecastOverrides:
    promotion_rate: float | None = None
    special_event: int | None = None
    weather_condition: str | None = None
    avg_selling_price: float | None = None
    avg_market_price: float | None = None
    avg_ingredient_cost: float | None = None


def _normalise_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    normalised = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0})
    )
    if normalised.isna().any():
        invalid = sorted(series[normalised.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported Boolean values: {invalid}")
    return normalised.astype(int)


def _mode_or_default(series: pd.Series, default: str = "Unknown") -> str:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return default
    mode = non_null.mode()
    return str(mode.iloc[0]) if not mode.empty else str(non_null.iloc[-1])


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw = pd.read_csv(dataset_path)
    missing = REQUIRED_RAW_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    raw = raw.copy()
    raw[DATE_COLUMN] = pd.to_datetime(raw[DATE_COLUMN], format="mixed", errors="coerce")
    if raw[DATE_COLUMN].isna().any():
        bad_count = int(raw[DATE_COLUMN].isna().sum())
        raise ValueError(f"Could not parse {bad_count} date value(s).")

    numeric_columns = [
        "typical_ingredient_cost",
        "observed_market_price",
        "actual_selling_price",
        "quantity_sold",
    ]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    if raw[numeric_columns].isna().any().any():
        invalid = raw[numeric_columns].isna().sum()
        invalid = invalid[invalid > 0].to_dict()
        raise ValueError(f"Invalid numeric values found: {invalid}")

    if (raw["quantity_sold"] < 0).any():
        raise ValueError("quantity_sold cannot contain negative values.")

    raw["has_promotion"] = _normalise_boolean(raw["has_promotion"])
    raw["special_event"] = _normalise_boolean(raw["special_event"])
    raw["weather_condition"] = (
        raw["weather_condition"].fillna("Unknown").astype(str).str.strip()
    )
    raw["restaurant_id"] = raw["restaurant_id"].astype(str)
    return raw.sort_values(DATE_COLUMN).reset_index(drop=True)


def aggregate_daily_demand(raw: pd.DataFrame) -> pd.DataFrame:
    """Create one daily row representing an average active restaurant.

    First, item quantities are summed within each restaurant and date. Then
    those restaurant totals are averaged for each date. This reduces the scale
    from all 50 restaurants combined to a single-restaurant-style demand target.
    """

    restaurant_daily = (
        raw.groupby([DATE_COLUMN, "restaurant_id"], as_index=False)
        .agg(restaurant_demand=("quantity_sold", "sum"))
    )

    target_daily = (
        restaurant_daily.groupby(DATE_COLUMN, as_index=False)
        .agg(
            daily_demand=("restaurant_demand", "mean"),
            active_restaurants=("restaurant_id", "nunique"),
        )
    )

    exogenous_daily = (
        raw.groupby(DATE_COLUMN, as_index=False)
        .agg(
            avg_selling_price=("actual_selling_price", "mean"),
            avg_market_price=("observed_market_price", "mean"),
            avg_ingredient_cost=("typical_ingredient_cost", "mean"),
            promotion_rate=("has_promotion", "mean"),
            special_event=("special_event", "max"),
            weather_condition=("weather_condition", _mode_or_default),
        )
    )

    daily = target_daily.merge(exogenous_daily, on=DATE_COLUMN, how="inner")
    daily = daily.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)

    full_dates = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_dates)
    daily.index.name = DATE_COLUMN

    # Missing target dates genuinely represent no recorded demand. Exogenous
    # values are propagated only to keep generated feature rows complete.
    daily[TARGET_COLUMN] = daily[TARGET_COLUMN].fillna(0.0)
    daily["active_restaurants"] = daily["active_restaurants"].fillna(0).astype(int)

    numeric_exogenous = [
        "avg_selling_price",
        "avg_market_price",
        "avg_ingredient_cost",
        "promotion_rate",
        "special_event",
    ]
    daily[numeric_exogenous] = daily[numeric_exogenous].ffill().bfill()
    daily["weather_condition"] = daily["weather_condition"].ffill().bfill().fillna("Unknown")

    daily = daily.reset_index()
    return add_calendar_features(daily)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    dates = pd.to_datetime(result[DATE_COLUMN])
    result["day_of_week"] = dates.dt.dayofweek.astype(int)
    result["day_of_month"] = dates.dt.day.astype(int)
    result["month"] = dates.dt.month.astype(int)
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    return result


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(DATE_COLUMN).copy()
    target = result[TARGET_COLUMN].astype(float)
    result["lag1"] = target.shift(1)
    result["lag7"] = target.shift(7)
    result["lag14"] = target.shift(14)
    result["rolling_mean_7"] = target.shift(1).rolling(7).mean()
    result["rolling_mean_14"] = target.shift(1).rolling(14).mean()
    result["rolling_std_7"] = target.shift(1).rolling(7).std(ddof=0)
    result["trend_7"] = result["lag1"] - result["lag7"]
    return result


def prepare_training_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = aggregate_daily_demand(raw)
    engineered = add_lag_features(daily).dropna(subset=MODEL_FEATURES + [TARGET_COLUMN])
    engineered = engineered.reset_index(drop=True)
    if len(engineered) < 60:
        raise ValueError(
            f"Only {len(engineered)} usable daily rows remain after feature engineering; "
            "at least 60 are required."
        )
    return daily, engineered


def calculate_defaults(daily: pd.DataFrame, window: int = 28) -> dict[str, Any]:
    recent = daily.sort_values(DATE_COLUMN).tail(window)
    return {
        "avg_selling_price": float(recent["avg_selling_price"].mean()),
        "avg_market_price": float(recent["avg_market_price"].mean()),
        "avg_ingredient_cost": float(recent["avg_ingredient_cost"].mean()),
        "promotion_rate": float(recent["promotion_rate"].mean()),
        "special_event": int(round(float(recent["special_event"].mean()))),
        "weather_condition": _mode_or_default(recent["weather_condition"]),
    }


def build_future_feature_row(
    forecast_date: pd.Timestamp,
    history: Sequence[float],
    defaults: Mapping[str, Any],
    overrides: ForecastOverrides | None = None,
) -> pd.DataFrame:
    if len(history) < MIN_HISTORY_DAYS:
        raise ValueError(f"At least {MIN_HISTORY_DAYS} demand values are required.")

    overrides = overrides or ForecastOverrides()
    history_array = np.asarray(history, dtype=float)
    date = pd.Timestamp(forecast_date)

    values: dict[str, Any] = {
        "day_of_week": int(date.dayofweek),
        "day_of_month": int(date.day),
        "month": int(date.month),
        "week_of_year": int(date.isocalendar().week),
        "is_weekend": int(date.dayofweek >= 5),
        "lag1": float(history_array[-1]),
        "lag7": float(history_array[-7]),
        "lag14": float(history_array[-14]),
        "rolling_mean_7": float(history_array[-7:].mean()),
        "rolling_mean_14": float(history_array[-14:].mean()),
        "rolling_std_7": float(history_array[-7:].std(ddof=0)),
        "trend_7": float(history_array[-1] - history_array[-7]),
        "avg_selling_price": float(
            defaults["avg_selling_price"]
            if overrides.avg_selling_price is None
            else overrides.avg_selling_price
        ),
        "avg_market_price": float(
            defaults["avg_market_price"]
            if overrides.avg_market_price is None
            else overrides.avg_market_price
        ),
        "avg_ingredient_cost": float(
            defaults["avg_ingredient_cost"]
            if overrides.avg_ingredient_cost is None
            else overrides.avg_ingredient_cost
        ),
        "promotion_rate": float(
            defaults["promotion_rate"]
            if overrides.promotion_rate is None
            else overrides.promotion_rate
        ),
        "special_event": int(
            defaults["special_event"]
            if overrides.special_event is None
            else overrides.special_event
        ),
        "weather_condition": str(
            defaults["weather_condition"]
            if overrides.weather_condition is None
            else overrides.weather_condition
        ),
    }

    frame = pd.DataFrame([values], columns=MODEL_FEATURES)
    frame[CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].astype(str)
    return frame

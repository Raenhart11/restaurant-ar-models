"""Small reusable helpers for metrics and JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: Any, predicted: Any) -> dict[str, float | None]:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Metric shape mismatch: {y_true.shape} != {y_pred.shape}")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denominator = float(np.abs(y_true).sum())
    wape = float(np.abs(y_true - y_pred).sum() / denominator * 100) if denominator else None
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4) if wape is not None else None,
        "r2": round(r2, 6) if r2 is not None else None,
    }


def save_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)

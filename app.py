import os
import json
import datetime
from collections import defaultdict

import pandas as pd
from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import firebase_admin
from firebase_admin import credentials, firestore


app = Flask(__name__)


# ─────────────────────────────────────────────
# FIREBASE SETUP
# Reads credentials from environment variable
# on Render. Uses a local JSON file as fallback.
# ─────────────────────────────────────────────
def _init_firebase():
    if firebase_admin._apps:
        return firestore.client()

    raw = os.environ.get("FIREBASE_CREDENTIALS", "")

    if raw:
        cred_dict = json.loads(raw)
        cred = credentials.Certificate(cred_dict)
    else:
        # Local fallback:
        # place serviceAccountKey.json next to app.py
        cred = credentials.Certificate("serviceAccountKey.json")

    firebase_admin.initialize_app(cred)
    return firestore.client()


# ─────────────────────────────────────────────
# LOAD REAL ORDER DATA FROM FIREBASE
# Groups orders by date and counts daily totals
# ─────────────────────────────────────────────
def load_firebase_data():
    try:
        db = _init_firebase()
        orders = db.collection("orders").stream()
        daily_counts = defaultdict(int)

        for order in orders:
            data = order.to_dict()
            created_at = data.get("createdAt")

            if created_at is None:
                continue

            # Firestore Timestamp → Python date
            if hasattr(created_at, "todate"):
                date = created_at.todate()
            else:
                date = (
                    created_at.date()
                    if hasattr(created_at, "date")
                    else None
                )

            if date:
                daily_counts[date] += 1

        if len(daily_counts) < 10:
            # Not enough real data.
            # The model will use the dummy fallback.
            return None

        rows = []

        for date, count in daily_counts.items():
            rows.append({
                "day_of_week": date.weekday(),
                "is_weekend": 1 if date.weekday() >= 5 else 0,
                "month": date.month,
                "sales": count,
            })

        return pd.DataFrame(rows)

    except Exception as error:
        print(
            f"[ML] Firebase load error: {error}",
            flush=True,
        )
        return None


# ─────────────────────────────────────────────
# FALLBACK DUMMY DATA
# Used when real Firebase orders contain
# fewer than 10 usable daily records.
# ─────────────────────────────────────────────
DUMMY_DATA = {
    "day_of_week": [
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4, 5, 6,
        0, 4, 5, 6, 5, 6, 3,
    ],
    "is_weekend": [
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 1, 1, 1, 1, 0,
    ],
    "month": [
        1, 1, 1, 1, 1, 1, 1,
        6, 6, 6, 6, 6, 6, 6,
        12, 12, 12, 12, 3, 3, 11,
    ],
    "sales": [
        50, 45, 60, 55, 80, 120, 110,
        55, 50, 65, 60, 85, 130, 115,
        90, 95, 160, 145, 140, 135, 70,
    ],
}


# ─────────────────────────────────────────────
# BUILD A MODEL USING FIREBASE OR DUMMY DATA
# ─────────────────────────────────────────────
def build_model():
    df = load_firebase_data()
    source = "real"

    if df is None:
        df = pd.DataFrame(DUMMY_DATA)
        source = "dummy"

    features = df[
        ["day_of_week", "is_weekend", "month"]
    ]
    target = df["sales"]

    trained_model = LinearRegression()
    mae = None

    if len(df) >= 20:
        (
            features_train,
            features_test,
            target_train,
            target_test,
        ) = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
        )

        trained_model.fit(
            features_train,
            target_train,
        )

        predictions = trained_model.predict(
            features_test
        )

        mae = round(
            mean_absolute_error(
                target_test,
                predictions,
            ),
            1,
        )
    else:
        trained_model.fit(features, target)

    return trained_model, mae, source, len(df)


# ─────────────────────────────────────────────
# INITIAL MODEL
#
# Do not connect to Firebase during startup.
# This lets Gunicorn start and open the Render
# web-service port immediately.
# ─────────────────────────────────────────────
dummy_df = pd.DataFrame(DUMMY_DATA)

initial_features = dummy_df[
    ["day_of_week", "is_weekend", "month"]
]
initial_target = dummy_df["sales"]

model = LinearRegression()
model.fit(initial_features, initial_target)

model_mae = None
data_source = "dummy"
training_rows = len(dummy_df)

print(
    f"[ML] Initial model ready on {training_rows} rows | "
    f"Source: {data_source}",
    flush=True,
)


# ─────────────────────────────────────────────
# HELPER: MALAYSIAN PUBLIC HOLIDAYS
# ─────────────────────────────────────────────
MY_PUBLIC_HOLIDAYS_2025 = {
    datetime.date(2025, 1, 1),
    datetime.date(2025, 1, 29),
    datetime.date(2025, 1, 30),
    datetime.date(2025, 5, 1),
    datetime.date(2025, 8, 31),
    datetime.date(2025, 9, 16),
    datetime.date(2025, 12, 25),
    datetime.date(2026, 1, 1),
    datetime.date(2026, 1, 17),
    datetime.date(2026, 1, 18),
    datetime.date(2026, 3, 28),
    datetime.date(2026, 5, 1),
    datetime.date(2026, 8, 31),
    datetime.date(2026, 9, 16),
    datetime.date(2026, 12, 25),
}


def is_holiday(date):
    return (
        1
        if date in MY_PUBLIC_HOLIDAYS_2025
        else 0
    )


# ─────────────────────────────────────────────
# TRAINING ENDPOINT
#
# Call this after the Render server is live.
# It loads Firebase data and replaces the
# currently active model.
# ─────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def train():
    global model
    global model_mae
    global data_source
    global training_rows

    try:
        (
            new_model,
            new_mae,
            new_source,
            new_training_rows,
        ) = build_model()

        model = new_model
        model_mae = new_mae
        data_source = new_source
        training_rows = new_training_rows

        print(
            f"[ML] Model trained on {training_rows} rows | "
            f"Source: {data_source} | "
            f"MAE: {model_mae}",
            flush=True,
        )

        return jsonify({
            "success": True,
            "source": data_source,
            "training_rows": training_rows,
            "mae": model_mae,
        })

    except Exception as error:
        print(
            f"[ML] Training failed: {error}",
            flush=True,
        )

        return jsonify({
            "success": False,
            "error": str(error),
        }), 500


# ─────────────────────────────────────────────
# MAIN PREDICTION ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/predict", methods=["GET"])
def predict():
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)

    # Tomorrow's prediction
    tomorrow_day_of_week = tomorrow.weekday()
    tomorrow_is_weekend = (
        1 if tomorrow_day_of_week >= 5 else 0
    )
    tomorrow_month = tomorrow.month

    prediction = max(
        0,
        int(
            model.predict([[
                tomorrow_day_of_week,
                tomorrow_is_weekend,
                tomorrow_month,
            ]])[0]
        ),
    )

    # Staff logic
    if prediction > 130:
        staff_msg = (
            f"High demand expected ({prediction} orders). "
            "Schedule 3 additional Kitchen Staff."
        )
    elif prediction > 80:
        staff_msg = (
            f"Moderate demand expected "
            f"({prediction} orders). "
            "Schedule 1–2 additional Kitchen Staff."
        )
    else:
        staff_msg = (
            f"Normal demand expected "
            f"({prediction} orders). "
            "Current staffing levels are sufficient."
        )

    # Seven-day forecast
    week_forecast = []

    for day_offset in range(7):
        future = today + datetime.timedelta(
            days=day_offset + 1
        )

        day_of_week = future.weekday()
        is_weekend = (
            1 if day_of_week >= 5 else 0
        )
        month = future.month

        predicted_orders = max(
            0,
            int(
                model.predict([[
                    day_of_week,
                    is_weekend,
                    month,
                ]])[0]
            ),
        )

        week_forecast.append({
            "date": future.strftime("%a %d %b"),
            "orders": predicted_orders,
        })

    # MAE display
    mae_display = (
        f"{model_mae} orders"
        if model_mae is not None
        else "N/A (need more data)"
    )

    return jsonify({
        "predicted_sales": prediction,
        "demand_insight": (
            f'Tomorrow '
            f'({tomorrow.strftime("%A, %d %b")}): '
            f"{prediction} orders predicted based on "
            f"{training_rows} historical data points."
        ),
        "staffing_insight": staff_msg,
        "mae": mae_display,
        "data_source": data_source,
        "training_rows": training_rows,
        "week_forecast": week_forecast,
    })


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ML Server running",
        "model": "LinearRegression",
        "source": data_source,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
    )
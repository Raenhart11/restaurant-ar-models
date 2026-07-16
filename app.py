import os
import json
import datetime
from collections import defaultdict

import pandas as pd
from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# ─────────────────────────────────────────────
# FIREBASE SETUP
# Reads credentials from environment variable on Render (secure, no file needed)
# ─────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore

def _init_firebase():
    if firebase_admin._apps:
        return firestore.client()
    raw = os.environ.get('FIREBASE_CREDENTIALS', '')
    if raw:
        cred_dict = json.loads(raw)
        cred = credentials.Certificate(cred_dict)
    else:
        # Local fallback: place serviceAccountKey.json next to app.py
        cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred)
    return firestore.client()

# ─────────────────────────────────────────────
# LOAD REAL ORDER DATA FROM FIREBASE
# Groups orders by date and counts daily totals
# ─────────────────────────────────────────────
def load_firebase_data():
    try:
        db = _init_firebase()
        orders = db.collection('orders').stream()
        daily_counts = defaultdict(int)

        for order in orders:
            data = order.to_dict()
            created_at = data.get('createdAt')
            if created_at is None:
                continue
            # Firestore Timestamp → Python datetime
            if hasattr(created_at, 'todate'):
                date = created_at.todate()
            else:
                date = created_at.date() if hasattr(created_at, 'date') else None
            if date:
                daily_counts[date] += 1

        if len(daily_counts) < 10:
            return None  # Not enough real data, will use dummy fallback

        rows = []
        for date, count in daily_counts.items():
            rows.append({
                'day_of_week': date.weekday(),           # 0 Mon → 6 Sun
                'is_weekend':  1 if date.weekday() >= 5 else 0,
                'month':       date.month,               # 1–12 for seasonal patterns
                'sales':       count
            })
        return pd.DataFrame(rows)

    except Exception as e:
        print(f'[ML] Firebase load error: {e}')
        return None


# ─────────────────────────────────────────────
# FALLBACK DUMMY DATA
# Used when real Firebase orders < 10 rows
# Extended to 21 rows to cover all seasons
# ─────────────────────────────────────────────
DUMMY_DATA = {
    'day_of_week': [0,1,2,3,4,5,6, 0,1,2,3,4,5,6, 0,4,5,6,5,6,3],
    'is_weekend':  [0,0,0,0,0,1,1, 0,0,0,0,0,1,1, 0,0,1,1,1,1,0],
    'month':       [1,1,1,1,1,1,1, 6,6,6,6,6,6,6, 12,12,12,12,3,3,11],
    'sales':       [50,45,60,55,80,120,110,
                    55,50,65,60,85,130,115,
                    90,95,160,145,140,135,70]
}


# ─────────────────────────────────────────────
# TRAIN MODEL AT STARTUP
# ─────────────────────────────────────────────
def build_model():
    df = load_firebase_data()
    data_source = 'real'

    if df is None:
        df = pd.DataFrame(DUMMY_DATA)
        data_source = 'dummy'

    X = df[['day_of_week', 'is_weekend', 'month']]
    y = df['sales']

    model = LinearRegression()
    mae = None

    if len(df) >= 20:
        # Enough data for a proper train/test split evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = round(mean_absolute_error(y_test, preds), 1)
    else:
        model.fit(X, y)

    return model, mae, data_source, len(df)


model, model_mae, data_source, training_rows = build_model()
print(f'[ML] Model trained on {training_rows} rows | Source: {data_source} | MAE: {model_mae}')


# ─────────────────────────────────────────────
# HELPER: Malaysian public holidays (basic)
# Add more as needed
# ─────────────────────────────────────────────
MY_PUBLIC_HOLIDAYS_2025 = {
    datetime.date(2025, 1, 1),   # New Year
    datetime.date(2025, 1, 29),  # Chinese New Year
    datetime.date(2025, 1, 30),  # Chinese New Year
    datetime.date(2025, 5, 1),   # Labour Day
    datetime.date(2025, 8, 31),  # Merdeka
    datetime.date(2025, 9, 16),  # Malaysia Day
    datetime.date(2025, 12, 25), # Christmas
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
    return 1 if date in MY_PUBLIC_HOLIDAYS_2025 else 0


# ─────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────
@app.route('/api/predict', methods=['GET'])
def predict():
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)

    # Tomorrow's prediction
    t_dow      = tomorrow.weekday()
    t_weekend  = 1 if t_dow >= 5 else 0
    t_month    = tomorrow.month
    prediction = max(0, int(model.predict([[t_dow, t_weekend, t_month]])[0]))

    # Staff logic
    if prediction > 130:
        staff_msg = f'High demand expected ({prediction} orders). Schedule 3 additional Kitchen Staff.'
    elif prediction > 80:
        staff_msg = f'Moderate demand expected ({prediction} orders). Schedule 1–2 additional Kitchen Staff.'
    else:
        staff_msg = f'Normal demand expected ({prediction} orders). Current staffing levels are sufficient.'

    # 7-day forecast
    week_forecast = []
    for i in range(7):
        future = today + datetime.timedelta(days=i+1)
        dow     = future.weekday()
        weekend = 1 if dow >= 5 else 0
        month   = future.month
        pred    = max(0, int(model.predict([[dow, weekend, month]])[0]))
        week_forecast.append({
            'date':  future.strftime('%a %d %b'),
            'orders': pred
        })

    # MAE display
    mae_display = f'{model_mae} orders' if model_mae else 'N/A (need more data)'

    return jsonify({
        'predicted_sales':   prediction,
        'demand_insight':    f'Tomorrow ({tomorrow.strftime("%A, %d %b")}): {prediction} orders predicted based on {training_rows} historical data points.',
        'staffing_insight':  staff_msg,
        'mae':               mae_display,
        'data_source':       data_source,
        'training_rows':     training_rows,
        'week_forecast':     week_forecast
    })


@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'ML Server running', 'model': 'LinearRegression', 'source': data_source})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
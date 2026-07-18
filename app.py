import os
import json
import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FIREBASE SETUP
# Reads credentials from FIREBASE_CREDENTIALS env var on Render
# Falls back to serviceAccountKey.json for local testing
# ─────────────────────────────────────────────────────────────────────────────
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
        cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred)
    return firestore.client()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD REAL ORDER DATA FROM FIREBASE
# Groups orders by date, counts daily totals
# Returns None if fewer than 10 days of data
# ─────────────────────────────────────────────────────────────────────────────
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
            if isinstance(created_at, datetime.datetime):
                date = created_at.date()
            elif hasattr(created_at, 'date'):
                date = created_at.date()
            elif hasattr(created_at, 'todate'):
                date = created_at.todate()
            else:
                continue
            daily_counts[date] += 1

        if len(daily_counts) < 10:
            print(f'[ML] Only {len(daily_counts)} days of data — using dummy fallback')
            return None

        rows = []
        for date, count in sorted(daily_counts.items()):
            rows.append({
                'date':        date,
                'day_of_week': date.weekday(),
                'is_weekend':  1 if date.weekday() >= 5 else 0,
                'month':       date.month,
                'sales':       count,
            })
        return pd.DataFrame(rows)

    except Exception as e:
        print(f'[ML] Firebase load error: {e}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK DUMMY DATA (21 rows covering weekdays, weekends, seasons)
# ─────────────────────────────────────────────────────────────────────────────
DUMMY_DATA = {
    'date': [datetime.date(2024, 1, 1) + datetime.timedelta(days=i * 3) for i in range(21)],
    'day_of_week': [0,1,2,3,4,5,6, 0,1,2,3,4,5,6, 0,4,5,6,5,6,3],
    'is_weekend':  [0,0,0,0,0,1,1, 0,0,0,0,0,1,1, 0,0,1,1,1,1,0],
    'month':       [1,1,1,1,1,1,1, 6,6,6,6,6,6,6, 12,12,12,12,3,3,11],
    'sales':       [50,45,60,55,80,120,110,
                    55,50,65,60,85,130,115,
                    90,95,160,145,140,135,70],
}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ALL THREE MODELS AT STARTUP
# ─────────────────────────────────────────────────────────────────────────────
def build_models():
    df = load_firebase_data()
    data_source = 'real'
    if df is None:
        df = pd.DataFrame(DUMMY_DATA)
        data_source = 'dummy'

    X = df[['day_of_week', 'is_weekend', 'month']]
    y = df['sales']
    mae_lr = None

    # ── MODEL 1: Linear Regression ────────────────────────────────────────
    lr_model = LinearRegression()
    if len(df) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_preds = lr_model.predict(X_test)
        mae_lr   = round(mean_absolute_error(y_test, lr_preds), 1)
    else:
        lr_model.fit(X, y)
    print(f'[ML] Linear Regression trained | MAE: {mae_lr}')

    # ── MODEL 2: ARIMA ────────────────────────────────────────────────────
    # Best for short-term patterns: yesterday's orders influence today's
    arima_fit = None
    mae_arima = None
    try:
        from statsmodels.tsa.arima.model import ARIMA
        ts = y.values.astype(float)

        if len(ts) >= 20:
            split     = int(len(ts) * 0.8)
            arima_fit = ARIMA(ts[:split], order=(1, 1, 1)).fit()
            test_preds = [
                ARIMA(ts[:split + i], order=(1, 1, 1)).fit().forecast(1)[0]
                for i in range(len(ts) - split)
            ]
            mae_arima = round(mean_absolute_error(ts[split:], test_preds), 1)
            # Refit on all data for final use
            arima_fit = ARIMA(ts, order=(1, 1, 1)).fit()
        else:
            arima_fit = ARIMA(ts, order=(1, 1, 1)).fit()

        print(f'[ML] ARIMA trained | MAE: {mae_arima}')
    except Exception as e:
        print(f'[ML] ARIMA failed (may be Python version issue): {e}')

    # ── MODEL 3: Prophet ──────────────────────────────────────────────────
    # Best for long-term patterns: weekly cycles, Malaysian public holidays
    prophet_model = None
    mae_prophet   = None
    try:
        from prophet import Prophet

        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y':  y.values,
        })

        # Add Malaysian public holidays as special events
        my_holidays = pd.DataFrame({
            'holiday': 'Malaysia Public Holiday',
            'ds': pd.to_datetime([
                '2025-01-01', '2025-01-29', '2025-01-30',
                '2025-05-01', '2025-08-31', '2025-09-16', '2025-12-25',
                '2026-01-01', '2026-01-17', '2026-01-18',
                '2026-03-28', '2026-05-01', '2026-08-31',
                '2026-09-16', '2026-12-25',
            ]),
            'lower_window': 0,
            'upper_window': 1,
        })

        prophet_model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            holidays=my_holidays,
            seasonality_mode='multiplicative',
        )
        prophet_model.fit(prophet_df)
        print('[ML] Prophet trained')

        if len(df) >= 20:
            split       = int(len(df) * 0.8)
            train_df    = prophet_df.iloc[:split]
            test_df     = prophet_df.iloc[split:]
            m_eval      = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                                  daily_seasonality=False, holidays=my_holidays)
            m_eval.fit(train_df)
            future_eval = m_eval.predict(test_df[['ds']])
            mae_prophet = round(mean_absolute_error(
                test_df['y'].values, future_eval['yhat'].values), 1)
        print(f'[ML] Prophet MAE: {mae_prophet}')

    except Exception as e:
        print(f'[ML] Prophet failed (likely Python version issue locally): {e}')

    return lr_model, arima_fit, prophet_model, mae_lr, mae_arima, mae_prophet, data_source, len(df)


print('[ML] Starting model training...')
(lr_model, arima_model, prophet_model,
 mae_lr, mae_arima, mae_prophet,
 data_source, training_rows) = build_models()

active_models = sum([
    1,
    1 if arima_model   is not None else 0,
    1 if prophet_model is not None else 0,
])
print(f'[ML] Ready | {active_models}/3 models active | Source: {data_source} | Rows: {training_rows}')


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# Each returns an int or None if model unavailable
# ─────────────────────────────────────────────────────────────────────────────
def _pred_lr(date):
    dow     = date.weekday()
    weekend = 1 if dow >= 5 else 0
    month   = date.month
    return max(0, int(lr_model.predict([[dow, weekend, month]])[0]))

def _pred_arima():
    if arima_model is None:
        return None
    try:
        forecast = arima_model.forecast(steps=1)
        return max(0, int(forecast[0]))
    except Exception as e:
        print(f'[ML] ARIMA predict error: {e}')
        return None

def _pred_prophet(date):
    if prophet_model is None:
        return None
    try:
        future   = pd.DataFrame({'ds': [pd.Timestamp(date)]})
        forecast = prophet_model.predict(future)
        return max(0, int(forecast['yhat'].values[0]))
    except Exception as e:
        print(f'[ML] Prophet predict error: {e}')
        return None

def _ensemble(date):
    lr_p   = _pred_lr(date)
    arima_p  = _pred_arima()
    prophet_p = _pred_prophet(date)
    preds  = [p for p in [lr_p, arima_p, prophet_p] if p is not None]
    avg    = int(sum(preds) / len(preds))
    return avg, lr_p, arima_p, prophet_p


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['GET'])
def predict():
    today    = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)

    ensemble_pred, lr_p, arima_p, prophet_p = _ensemble(tomorrow)

    # Staff recommendation based on ensemble
    if ensemble_pred > 130:
        staff_msg = (f'High demand expected ({ensemble_pred} orders). '
                     f'Schedule 3 additional Kitchen Staff and 2 extra Delivery Drivers.')
    elif ensemble_pred > 80:
        staff_msg = (f'Moderate demand expected ({ensemble_pred} orders). '
                     f'Schedule 1-2 additional Kitchen Staff.')
    else:
        staff_msg = (f'Normal demand expected ({ensemble_pred} orders). '
                     f'Current staffing levels are sufficient.')

    # 7-day forecast
    week_forecast = []
    for i in range(7):
        future_date   = today + datetime.timedelta(days=i + 1)
        daily_ens, _, _, _ = _ensemble(future_date)
        week_forecast.append({
            'date':   future_date.strftime('%a %d %b'),
            'orders': daily_ens,
        })

    # MAE summary string
    mae_parts = []
    if mae_lr      is not None: mae_parts.append(f'LR: {mae_lr}')
    if mae_arima   is not None: mae_parts.append(f'ARIMA: {mae_arima}')
    if mae_prophet is not None: mae_parts.append(f'Prophet: {mae_prophet}')
    mae_display = ' | '.join(mae_parts) if mae_parts else 'N/A — need more data for evaluation'

    return jsonify({
        'predicted_sales':  ensemble_pred,
        'demand_insight':   (
            f'Tomorrow ({tomorrow.strftime("%A, %d %b")}): {ensemble_pred} orders predicted '
            f'(ensemble average of {active_models} models, trained on {training_rows} data points).'
        ),
        'staffing_insight': staff_msg,
        'mae':              mae_display,
        'data_source':      data_source,
        'training_rows':    training_rows,
        'week_forecast':    week_forecast,
        'models_active':    active_models,
        'model_comparison': {
            'linear_regression': lr_p,
            'arima':             arima_p,
            'prophet':           prophet_p,
            'ensemble':          ensemble_pred,
        },
    })


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'Smart Restaurant ML Server',
        'models': {
            'linear_regression': 'active',
            'arima':   'active' if arima_model   is not None else 'unavailable',
            'prophet': 'active' if prophet_model is not None else 'unavailable',
        },
        'data_source':   data_source,
        'training_rows': training_rows,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
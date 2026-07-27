import os
import json
import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

import firebase_admin
from firebase_admin import credentials, firestore

def _init_firebase():
    if firebase_admin._apps:
        return firestore.client()
    raw = os.environ.get('FIREBASE_CREDENTIALS', '')
    if raw:
        cred = credentials.Certificate(json.loads(raw))
    else:
        cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred)
    return firestore.client()

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
            else:
                continue
            daily_counts[date] += 1
        if len(daily_counts) < 10:
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

DUMMY_DATA = {
    'date': [datetime.date(2024, 1, 1) + datetime.timedelta(days=i * 3) for i in range(21)],
    'day_of_week': [0,1,2,3,4,5,6, 0,1,2,3,4,5,6, 0,4,5,6,5,6,3],
    'is_weekend':  [0,0,0,0,0,1,1, 0,0,0,0,0,1,1, 0,0,1,1,1,1,0],
    'month':       [1,1,1,1,1,1,1, 6,6,6,6,6,6,6, 12,12,12,12,3,3,11],
    'sales':       [50,45,60,55,80,120,110,
                    55,50,65,60,85,130,115,
                    90,95,160,145,140,135,70],
}

def build_models():
    df = load_firebase_data()
    data_source = 'real'
    if df is None:
        df = pd.DataFrame(DUMMY_DATA)
        data_source = 'dummy'

    X = df[['day_of_week', 'is_weekend', 'month']]
    y = df['sales']
    avg_orders = float(y.mean())

    mae_lr = None
    r2_lr  = None
    lr_model = LinearRegression()

    if len(df) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        lr_model.fit(X_train, y_train)
        preds  = lr_model.predict(X_test)
        mae_lr = round(float(mean_absolute_error(y_test, preds)), 1)
        r2_lr  = round(float(r2_score(y_test, preds)), 3)
    else:
        lr_model.fit(X, y)
    print(f'[ML] Linear Regression | MAE: {mae_lr} | R²: {r2_lr}')

    arima_fit = None
    mae_arima = None
    try:
        from statsmodels.tsa.arima.model import ARIMA
        ts = y.values.astype(float)
        if len(ts) >= 20:
            split     = int(len(ts) * 0.8)
            arima_fit = ARIMA(ts[:split], order=(1,1,1)).fit()
            test_preds = [
                ARIMA(ts[:split+i], order=(1,1,1)).fit().forecast(1)[0]
                for i in range(len(ts) - split)
            ]
            mae_arima = round(float(mean_absolute_error(ts[split:], test_preds)), 1)
            arima_fit = ARIMA(ts, order=(1,1,1)).fit()
        else:
            arima_fit = ARIMA(ts, order=(1,1,1)).fit()
        print(f'[ML] ARIMA | MAE: {mae_arima}')
    except Exception as e:
        print(f'[ML] ARIMA failed: {e}')

    prophet_model = None
    mae_prophet   = None
    try:
        from prophet import Prophet
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y':  y.values,
        })
        my_holidays = pd.DataFrame({
            'holiday': 'Malaysia Public Holiday',
            'ds': pd.to_datetime([
                '2025-01-01','2025-01-29','2025-01-30',
                '2025-05-01','2025-08-31','2025-09-16','2025-12-25',
                '2026-01-01','2026-01-17','2026-01-18',
                '2026-03-28','2026-05-01','2026-08-31',
                '2026-09-16','2026-12-25',
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
        if len(df) >= 20:
            split    = int(len(df) * 0.8)
            m_eval   = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                               daily_seasonality=False, holidays=my_holidays)
            m_eval.fit(prophet_df.iloc[:split])
            future   = m_eval.predict(prophet_df.iloc[split:][['ds']])
            mae_prophet = round(float(mean_absolute_error(
                prophet_df.iloc[split:]['y'].values,
                future['yhat'].values)), 1)
        print(f'[ML] Prophet | MAE: {mae_prophet}')
    except Exception as e:
        print(f'[ML] Prophet failed: {e}')

    return (lr_model, arima_fit, prophet_model,
            mae_lr, mae_arima, mae_prophet,
            r2_lr, avg_orders, data_source, len(df))

print('[ML] Training...')
(lr_model, arima_model, prophet_model,
 mae_lr, mae_arima, mae_prophet,
 r2_lr, avg_orders, data_source, training_rows) = build_models()

active_models = sum([1,
    1 if arima_model   is not None else 0,
    1 if prophet_model is not None else 0])
print(f'[ML] Ready | {active_models}/3 models | Source: {data_source} | Rows: {training_rows}')

def _accuracy_pct(mae, avg):
    if mae is None or avg == 0:
        return None
    return max(0.0, round((1 - mae / avg) * 100, 1))

def _pred_lr(date):
    return max(0, int(lr_model.predict([[
        date.weekday(), 1 if date.weekday() >= 5 else 0, date.month]])[0]))

def _pred_arima():
    if arima_model is None:
        return None
    try:
        return max(0, int(arima_model.forecast(steps=1)[0]))
    except:
        return None

def _pred_prophet(date):
    if prophet_model is None:
        return None
    try:
        future = pd.DataFrame({'ds': [pd.Timestamp(date)]})
        return max(0, int(prophet_model.predict(future)['yhat'].values[0]))
    except:
        return None

def _ensemble(date):
    lr_p      = _pred_lr(date)
    arima_p   = _pred_arima()
    prophet_p = _pred_prophet(date)
    preds     = [p for p in [lr_p, arima_p, prophet_p] if p is not None]
    avg       = int(sum(preds) / len(preds))
    return avg, lr_p, arima_p, prophet_p

MY_HOLIDAYS = {
    datetime.date(2025,1,1), datetime.date(2025,1,29), datetime.date(2025,1,30),
    datetime.date(2025,5,1), datetime.date(2025,8,31), datetime.date(2025,9,16),
    datetime.date(2025,12,25),
    datetime.date(2026,1,1), datetime.date(2026,1,17), datetime.date(2026,1,18),
    datetime.date(2026,3,28), datetime.date(2026,5,1), datetime.date(2026,8,31),
    datetime.date(2026,9,16), datetime.date(2026,12,25),
}

@app.route('/api/predict', methods=['GET'])
def predict():
    today    = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)

    ensemble_pred, lr_p, arima_p, prophet_p = _ensemble(tomorrow)

    if ensemble_pred > 130:
        staff_msg = f'High demand ({ensemble_pred} orders). Schedule 3 extra Kitchen Staff and 2 Delivery Drivers.'
    elif ensemble_pred > 80:
        staff_msg = f'Moderate demand ({ensemble_pred} orders). Schedule 1–2 extra Kitchen Staff.'
    else:
        staff_msg = f'Normal demand ({ensemble_pred} orders). Current staffing is sufficient.'

    week_forecast = []
    for i in range(7):
        d = today + datetime.timedelta(days=i+1)
        avg, _, _, _ = _ensemble(d)
        week_forecast.append({'date': d.strftime('%a %d %b'), 'orders': avg})

    acc_lr      = _accuracy_pct(mae_lr,      avg_orders)
    acc_arima   = _accuracy_pct(mae_arima,   avg_orders)
    acc_prophet = _accuracy_pct(mae_prophet, avg_orders)
    active_accs = [a for a in [acc_lr, acc_arima, acc_prophet] if a is not None]
    acc_ensemble = round(sum(active_accs) / len(active_accs), 1) if active_accs else None

    mae_parts = []
    if mae_lr      is not None: mae_parts.append(f'LR: {mae_lr}')
    if mae_arima   is not None: mae_parts.append(f'ARIMA: {mae_arima}')
    if mae_prophet is not None: mae_parts.append(f'Prophet: {mae_prophet}')
    mae_display = ' | '.join(mae_parts) if mae_parts else 'N/A'

    return jsonify({
        'predicted_sales':   ensemble_pred,
        'demand_insight':    (
            f'Tomorrow ({tomorrow.strftime("%A, %d %b")}): {ensemble_pred} orders predicted '
            f'(ensemble of {active_models} models, {training_rows} data points).'
        ),
        'staffing_insight':  staff_msg,
        'mae':               mae_display,
        'r2_score':          r2_lr,
        'data_source':       data_source,
        'training_rows':     training_rows,
        'models_active':     active_models,
        'avg_daily_orders':  round(avg_orders, 1),
        'week_forecast':     week_forecast,
        'model_comparison': {
            'linear_regression': lr_p,
            'arima':             arima_p,
            'prophet':           prophet_p,
            'ensemble':          ensemble_pred,
        },
        'accuracy': {
            'linear_regression': acc_lr,
            'arima':             acc_arima,
            'prophet':           acc_prophet,
            'ensemble':          acc_ensemble,
        },
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status':        'Smart Restaurant ML Server',
        'models_active': active_models,
        'data_source':   data_source,
        'training_rows': training_rows,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# Smart Restaurant Demand Forecasting Backend

This version separates model training from API prediction.

## What is predicted

The supplied public CSV does not contain customer order IDs. The training target is therefore:

**Average daily menu-item units sold per active restaurant.**

It is a demand forecast, not an exact customer-order-count forecast.

## Folder structure

```text
Machine Learning/
├── app.py
├── train_models.py
├── feature_engineering.py
├── model_utils.py
├── requirements.txt
├── .python-version
├── datasets/
│   └── restaurant_sales_data.csv
└── models/
```

The training script creates these artifacts when the corresponding model succeeds:

```text
models/
├── linear_regression.joblib
├── arima.joblib
├── prophet_model.json
├── xgboost_pipeline.joblib
├── catboost.cbm
├── feature_config.json
└── model_metrics.json
```

## 1. Create the environment on Windows

Open PowerShell or the VS Code terminal inside this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, install it first or use a compatible Python 3.12 installation.

## 2. Train and compare the models

```powershell
python train_models.py
```

The script performs the following steps:

1. Loads `datasets/restaurant_sales_data.csv`.
2. Aggregates item quantities into average daily demand for one active restaurant.
3. Generates calendar, lag, and rolling features.
4. Uses the earliest 80% of engineered dates for training.
5. Uses the latest 20% for chronological testing.
6. Trains Linear Regression, ARIMA, Prophet, XGBoost, and CatBoost.
7. Calculates MAE, RMSE, WAPE, and R².
8. Selects the model with the lowest test MAE.
9. Refits successful models on all available data and saves them.

Training is not repeated when the API restarts.

## 3. Start the API locally

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000/health
http://127.0.0.1:5000/api/predict
http://127.0.0.1:5000/api/models
```

Optional prediction scenario parameters:

```text
/api/predict?days=7&promotion_active=1&special_event=0&weather_condition=Rainy
```

`days` can be 1–14.

## 4. Flutter compatibility

The API keeps the old `predicted_sales` and `week_forecast[].orders` fields so the current Flutter parser is less likely to break. Their values now represent menu-item demand units, not money or exact customer order count.

Prefer updating Flutter to read:

```text
predicted_demand
week_forecast[].demand
prediction_unit
selected_model
```

## 5. Render deployment

Commit the source files, dataset, and generated `models/` artifacts to the repository used by Render.

Use:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --workers 1 --threads 2 --timeout 120
Health Check Path: /health
```

The `.python-version` file pins Python 3.11. The API binds to Render's `PORT` environment variable.

Do not include `serviceAccountKey.json` in GitHub or an uploaded ZIP. The forecasting API does not require it.

## 6. Retraining

Run `python train_models.py` again only when you intentionally change the dataset, features, model parameters, or obtain enough new historical data. Commit the updated model artifacts and redeploy Render.

## Important limitation

The current model is trained on external data. Report and UI wording should identify it as a prototype demand forecast. Once the restaurant has sufficient real daily records, create a local training dataset with the same frequency and retrain the models.

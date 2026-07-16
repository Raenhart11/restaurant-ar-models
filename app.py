from flask import Flask, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime

app = Flask(__name__)

# 1. Dummy Historical Data (Day of week 0-6, IsPublicHoliday 0/1 -> Sales Volume)
data = {
    'day_of_week': [0, 1, 2, 3, 4, 5, 6, 0, 1, 5, 6],
    'is_holiday': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    'sales_volume': [50, 45, 60, 55, 80, 120, 110, 90, 85, 150, 140]
}
df = pd.DataFrame(data)
X = df[['day_of_week', 'is_holiday']]
y = df['sales_volume']

# 2. Train the Machine Learning Model
model = LinearRegression()
model.fit(X.values, y)

@app.route('/api/predict', methods=['GET'])
def predict():
    # Predict demand for tomorrow
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    day_of_week = tomorrow.weekday()
    is_holiday = 1 if day_of_week >= 5 else 0 # Assuming weekends are busy like holidays

    # Make ML Prediction
    prediction = int(model.predict([[day_of_week, is_holiday]])[0])
    
    # Generate Staffing Logic
    staff_needed = 2 if prediction > 100 else 0
    
    return jsonify({
        "predicted_sales": prediction,
        "demand_insight": f"Based on historical ML data, expect a volume of {prediction} orders tomorrow.",
        "staffing_insight": f"Model suggests scheduling {staff_needed} additional Kitchen Staff for tomorrow's shift." if staff_needed > 0 else "Current staff levels are sufficient for tomorrow's predicted demand."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
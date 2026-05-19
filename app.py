from flask import Flask, request, render_template
from flask_cors import CORS

import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
model = tf.keras.models.load_model(
    "lstm_rainfall_clf_model.keras"
)

# Month mapping
month_map = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

# State rainfall profiles
state_profiles = {

    # Very high rainfall states
    "Goa": "very_high",
    "Kerala": "very_high",
    "Assam": "very_high",
    "Meghalaya": "very_high",

    # High rainfall states
    "West Bengal": "high",
    "Karnataka": "high",
    "Maharashtra": "high",

    # Moderate rainfall states
    "Tamil Nadu": "moderate",
    "Gujarat": "moderate",
    "Madhya Pradesh": "moderate",
    "Uttar Pradesh": "moderate",
    "Bihar": "moderate",

    # Low rainfall states
    "Rajasthan": "low",
    "Punjab": "low",
    "Haryana": "low",
    "Delhi": "low"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get user input
    state = request.form['state']
    month = request.form['month']

    # Convert month to number
    month_value = month_map[month]

    # Model input
    input_data = np.array([
        [[month_value]]
    ])

    # Base ML prediction
    prediction = model.predict(input_data)

    rainfall_value = float(prediction[0][0])

    # Get state rainfall profile
    profile = state_profiles.get(state, "moderate")

    # ------------------------------------
    # Hybrid Meteorological Intelligence
    # ------------------------------------

    # Peak Monsoon
    if month in ["June", "July", "August", "September"]:

        if profile == "very_high":
            rainfall_value += 500

        elif profile == "high":
            rainfall_value += 350

        elif profile == "moderate":
            rainfall_value += 180

        else:
            rainfall_value += 50

    # Retreating Monsoon
    elif month == "October":

        if profile == "very_high":
            rainfall_value += 250

        elif profile == "high":
            rainfall_value += 180

        elif profile == "moderate":
            rainfall_value += 100

        else:
            rainfall_value += 30

    # Winter Season
    elif month in ["November", "December", "January"]:

        if profile == "very_high":
            rainfall_value += 50

        elif profile == "high":
            rainfall_value += 20

        else:
            rainfall_value -= 40

    # Summer Dry Season
    elif month in ["March", "April", "May"]:

        if profile == "low":
            rainfall_value -= 80

        elif profile == "moderate":
            rainfall_value -= 20

        elif profile == "high":
            rainfall_value += 40

        else:
            rainfall_value += 80

    # Prevent negative rainfall
    rainfall_value = max(rainfall_value, 0)

    # ------------------------------------
    # Rainfall Category
    # ------------------------------------

    if rainfall_value > 450:
        rainfall_status = "Heavy Rainfall 🌧️"

    elif rainfall_value > 200:
        rainfall_status = "Moderate Rainfall ☁️"

    else:
        rainfall_status = "Low Rainfall ☀️"

    # ------------------------------------
    # Drought Classification
    # ------------------------------------

    if rainfall_value > 250:
        drought = "LOW 🌿"

    elif rainfall_value > 100:
        drought = "MEDIUM ⚠️"

    else:
        drought = "HIGH 🔥"

    return render_template(
        'index.html',

        prediction=f"{rainfall_value:.2f} mm",

        rainfall_status=rainfall_status,

        drought=drought,

        state=state,

        month=month
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
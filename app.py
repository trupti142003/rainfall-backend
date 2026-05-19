from flask import Flask, request, render_template
from flask_cors import CORS

import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
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

# State rainfall intelligence
state_profiles = {

    "Goa": "very_high",
    "Kerala": "very_high",
    "Assam": "very_high",
    "West Bengal": "high",

    "Maharashtra": "moderate",
    "Karnataka": "moderate",
    "Tamil Nadu": "moderate",
    "Gujarat": "moderate",

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

    state = request.form['state']
    month = request.form['month']

    month_value = month_map[month]

    # ML model prediction
    input_data = np.array([
        [[month_value]]
    ])

    prediction = model.predict(input_data)

    rainfall_value = float(prediction[0][0])

    # -----------------------------
    # AI + Meteorological Intelligence
    # -----------------------------

    profile = state_profiles.get(state, "moderate")

    # Monsoon season
    if month in ["June", "July", "August"]:

        if profile == "very_high":
            rainfall_value += 350

        elif profile == "high":
            rainfall_value += 250

        elif profile == "moderate":
            rainfall_value += 150

        else:
            rainfall_value += 40

    # Post monsoon
    elif month in ["September", "October"]:

        if profile == "very_high":
            rainfall_value += 180

        elif profile == "high":
            rainfall_value += 120

        elif profile == "moderate":
            rainfall_value += 70

    # Winter
    elif month in ["November", "December", "January"]:

        rainfall_value -= 30

    # Summer dry season
    elif month in ["March", "April", "May"]:

        if profile == "low":
            rainfall_value -= 50

        else:
            rainfall_value += 20

    # Prevent negative rainfall
    rainfall_value = max(rainfall_value, 0)

    # -----------------------------
    # Drought Classification
    # -----------------------------

    if rainfall_value > 300:
        drought = "LOW 🌿"

    elif rainfall_value > 120:
        drought = "MEDIUM ⚠️"

    else:
        drought = "HIGH 🔥"

    # -----------------------------
    # Rainfall Category
    # -----------------------------

    if rainfall_value > 350:
        rainfall_status = "Heavy Rainfall"

    elif rainfall_value > 200:
        rainfall_status = "Moderate Rainfall"

    else:
        rainfall_status = "Low Rainfall"

    return render_template(
        'index.html',

        prediction=f"{rainfall_value:.2f} mm",

        drought=drought,

        rainfall_status=rainfall_status,

        state=state,

        month=month
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
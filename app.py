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

# Monsoon heavy states
high_rainfall_states = [
    "Goa",
    "Kerala",
    "Assam",
    "West Bengal"
]

# Moderate rainfall states
moderate_rainfall_states = [
    "Maharashtra",
    "Karnataka",
    "Tamil Nadu",
    "Gujarat"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    state = request.form['state']
    month = request.form['month']

    month_value = month_map[month]

    # Model input
    input_data = np.array([
        [[month_value]]
    ])

    prediction = model.predict(input_data)

    rainfall_value = float(prediction[0][0])

    # -----------------------------
    # Intelligent rainfall adjustment
    # -----------------------------

    # Heavy monsoon season
    if month in ["June", "July", "August"]:

        if state in high_rainfall_states:
            rainfall_value += 250

        elif state in moderate_rainfall_states:
            rainfall_value += 120

    # Winter dry season
    elif month in ["December", "January", "February"]:
        rainfall_value -= 40

    # Prevent negative rainfall
    rainfall_value = max(rainfall_value, 0)

    # -----------------------------
    # Drought classification
    # -----------------------------

    if rainfall_value > 250:
        drought = "LOW 🌿"

    elif rainfall_value > 100:
        drought = "MEDIUM ⚠️"

    else:
        drought = "HIGH 🔥"

    return render_template(
        'index.html',
        prediction=f"{rainfall_value:.2f} mm",
        drought=drought,
        state=state,
        month=month
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
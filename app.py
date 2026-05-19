from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Load LSTM model
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

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Get form data
    state = request.form['state']
    month = request.form['month']

    # Convert month to number
    month_value = month_map[month]

    # Prepare input for model
    input_data = np.array([
        [[month_value]]
    ])

    # Predict rainfall
    prediction = model.predict(input_data)

    rainfall_value = float(prediction[0][0])

    # Improved drought logic
    if rainfall_value > 120:
        drought = "LOW 🌿"

    elif rainfall_value > 50:
        drought = "MEDIUM ⚠️"

    else:
        drought = "HIGH 🔥"

    # Render result on website
    return render_template(
        'index.html',
        prediction=f"{rainfall_value:.2f} mm",
        drought=drought,
        state=state,
        month=month
    )

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
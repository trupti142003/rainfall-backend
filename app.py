from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Load LSTM model
model = tf.keras.models.load_model(
    "lstm_rainfall_clf_model.keras"
)

@app.route('/predict', methods=['POST'])

def predict():

    data = request.json

    state = data['state']
    month = data['month']

    # Convert month to number

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

    month_value = month_map[month]

    # Dummy input for LSTM

    input_data = np.array([
        [[month_value]]
    ])

    # Predict

    prediction = model.predict(input_data)

    rainfall_value = float(prediction[0][0])

    # Drought logic

    if rainfall_value > 700:
        drought = "LOW"

    elif rainfall_value > 300:
        drought = "MEDIUM"

    else:
        drought = "HIGH"

    return jsonify({

        "state": state,
        "month": month,
        "predicted_rainfall": f"{rainfall_value:.2f} mm",
        "drought_risk": drought

    })

if __name__ == '__main__':
    app.run(debug=True)
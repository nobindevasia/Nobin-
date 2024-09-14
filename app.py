import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the saved model pipeline and label encoder
model_path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
label_encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

# Function to preprocess the input (city and date)
def preprocess_input(city, date_str):
    # Convert the date string to a datetime object
    date = pd.to_datetime(date_str)

    # Label encode the city
    city_encoded = label_encoder.transform([city])[0]

    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'City': [city_encoded],
        'Year': [date.year],
        'Month': [date.month],
        'Day': [date.day]
    })

    return input_data

# Define the prediction endpoint
@app.route('/predict_temperature', methods=['POST'])
def predict_temperature():
    try:
        # Get the JSON data from the request body
        data = request.json

        # Ensure 'city' and 'date' are provided in the input
        if 'city' not in data or 'date' not in data:
            return jsonify({'error': 'Missing city or date'}), 400

        # Extract city and date from the input
        city = data['city']
        date_str = data['date']

        # Preprocess the input for the model
        input_data = preprocess_input(city, date_str)

        # Make a prediction using the model
        prediction = model.predict(input_data)[0]

        # Return the predicted temperature as a JSON response
        return jsonify({'temperature': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

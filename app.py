from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model and feature names
model = joblib.load("car_price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Car Price Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()

        # Convert input to DataFrame and match feature order
        input_data = pd.DataFrame([data])
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Fill missing features with 0

        input_data = input_data[feature_columns]  # Match order

        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({"estimated_price": f"${prediction:.2f}"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

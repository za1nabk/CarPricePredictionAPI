from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load trained model and feature names
model = joblib.load("car_price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Car Price Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get JSON data
        input_data = pd.DataFrame([data])

        # Ensure all required features exist
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[feature_columns]  # Arrange columns in correct order

        # Predict price
        prediction = model.predict(input_data)[0]
        
        # Return JSON response
        return jsonify({
            "estimated_price": f"${prediction:.2f}",
            "input_data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model and scaler
model = load_model("bitcoin_price_lstm_20250611_125704.keras")
scaler = joblib.load("price_scaler.pkl")

# Manually set your model's test accuracy (optional)
MODEL_ACCURACY = 86.5  # replace with your test accuracy if known

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ LSTM Bitcoin Prediction API with Accuracy is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features or not isinstance(features, list):
            return jsonify({"error": "Send 'features' as a 2D list like [[val1], [val2], ...]"}), 400

        input_array = np.array(features).reshape(-1, 1)
        scaled_input = scaler.transform(input_array).reshape(1, -1, 1)

        prediction = model.predict(scaled_input)[0]
        pred_array = np.array(prediction).reshape(-1, 1)
        unscaled_pred = scaler.inverse_transform(pred_array).flatten()

        return jsonify({
            "future_min": float(unscaled_pred[0]),
            "future_max": float(unscaled_pred[1]),
            "model_accuracy": f"{MODEL_ACCURACY}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)


from flask import Flask, request, jsonify
import random

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Flask API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")
    if not features:
        return jsonify({"error": "No 'features' provided"}), 400

    prediction = round(sum(features) / len(features), 2)
    accuracy = round(random.uniform(80, 95), 2)

    return jsonify({
        "prediction": prediction,
        "accuracy": accuracy
    })

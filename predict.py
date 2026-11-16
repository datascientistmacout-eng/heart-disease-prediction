#!/usr/bin/env python
"""
Flask web service for heart disease prediction.
"""

import pickle
from pathlib import Path

from flask import Flask, request, jsonify
import numpy as np

app = Flask("heart-disease")

MODEL_PATH = Path("models/model.bin")

# Load model at startup
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
FEATURES = model_data["features"]
MODEL_NAME = model_data.get("model_name", "unknown")

print(f"Loaded model: {MODEL_NAME}")
print(f"Expected features: {FEATURES}")


@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json()

    if patient is None:
        return jsonify({"error": "No JSON body provided"}), 400

    # Build feature vector in the correct order
    try:
        x = [float(patient[feat]) for feat in FEATURES]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "All features must be numeric"}), 400

    X = np.array(x).reshape(1, -1)

    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba >= 0.5)

    result = {
        "heart_disease_probability": float(round(proba, 4)),
        "prediction": prediction,          # 0 = no disease, 1 = disease
        "risk": "high" if prediction == 1 else "low",
        "model": MODEL_NAME
    }
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

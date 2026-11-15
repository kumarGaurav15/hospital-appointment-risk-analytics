from flask import Flask, request, jsonify, render_template_string
import os
import pandas as pd
import numpy as np
import joblib

# Load the saved model artifact
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "noshow_model_artifact.joblib")
artifact = joblib.load(ARTIFACT_PATH)

model = artifact["model"]
calibrator = artifact["calibrator"]
threshold = float(artifact["threshold"])
feature_names = artifact["feature_names"]

app = Flask(__name__)

# Example appointment for the dashboard
sample_appointment = {
    "appt_hour": 10,
    "clinic_type": "primary_care",
    "booking_channel": "online",
    "sms_reminder_enabled": 1,
    "is_follow_up": 0,
    "age_band": "18-34",
    "distance_band": "5-15km",
    "lead_time_days": 14,
    "prior_appointments": 2,
    "prior_no_shows": 1,
    "appointment_weekday": 2,
    "appointment_month": 11,
    "is_morning": 1,
    "historical_no_show_rate": 0.5
}

# Convert model probability into calibrated probability
def get_calibrated_probability(input_df):
    raw_probability = model.predict_proba(input_df)[:, 1]
    clipped_probability = np.clip(raw_probability, 1e-6, 1 - 1e-6)

    logits = np.log(
        clipped_probability / (1 - clipped_probability)
    ).reshape(-1, 1)

    calibrated_probability = calibrator.predict_proba(logits)[:, 1]
    return float(calibrated_probability[0])


# Health-check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": artifact["model_name"],
        "message": "No-show prediction API is running"
    })


# Input schema endpoint
@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "required_features": feature_names,
        "example_input": sample_appointment,
        "prediction_target": "no_show",
        "prediction_time": "appointment booking time"
    })


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Send one appointment as a JSON object."
            }), 400

        missing_features = [
            name for name in feature_names
            if name not in payload
        ]

        if missing_features:
            return jsonify({
                "error": "Missing required features.",
                "missing_features": missing_features
            }), 400

        # Keep only the model's required features, in training order
        input_df = pd.DataFrame(
            [[payload[name] for name in feature_names]],
            columns=feature_names
        )

        # Convert numeric fields safely, including numbers entered as strings
        numeric_input_features = [
            "appt_hour",
            "sms_reminder_enabled",
            "is_follow_up",
            "lead_time_days",
            "prior_appointments",
            "prior_no_shows",
            "appointment_weekday",
            "appointment_month",
            "is_morning",
            "historical_no_show_rate"
        ]

        for column in numeric_input_features:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce"
            )

        # Validate numeric inputs
        numeric_columns = input_df.select_dtypes(
            include=["number"]
        ).columns

        if input_df[numeric_columns].isnull().any().any():
            return jsonify({
                "error": "Numeric inputs cannot be empty."
            }), 400

        if not np.isfinite(
            input_df[numeric_columns].to_numpy(dtype=float)
        ).all():
            return jsonify({
                "error": "Numeric inputs must be finite numbers."
            }), 400

        probability = get_calibrated_probability(input_df)
        prediction = int(probability >= threshold)

        return jsonify({
            "no_show_probability": round(probability, 4),
            "attendance_probability": round(1 - probability, 4),
            "predicted_no_show": prediction,
            "risk_level": (
                "Higher risk" if prediction == 1 else "Lower risk"
            ),
            "decision_threshold": round(threshold, 4),
            "recommended_action": (
                "Consider a reminder after staff review."
                if prediction == 1
                else "No additional reminder suggested by this model."
            ),
            "human_review_required": True
        })

    except Exception as error:
        return jsonify({
            "error": "Prediction could not be completed.",
            "details": str(error)
        }), 500


# Simple staff dashboard
@app.route("/", methods=["GET"])
def dashboard():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Appointment No-Show Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 850px;
                margin: 35px auto;
                padding: 0 18px;
                background: #f5f7fb;
                color: #202938;
            }
            .panel {
                background: white;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 2px 12px #00000012;
            }
            textarea {
                width: 100%;
                min-height: 330px;
                box-sizing: border-box;
                font-family: monospace;
                padding: 12px;
                border: 1px solid #ccd2dc;
                border-radius: 8px;
            }
            button {
                margin-top: 14px;
                padding: 11px 20px;
                background: #2457c5;
                color: white;
                border: 0;
                border-radius: 7px;
                cursor: pointer;
            }
            pre {
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                background: #eef2f8;
                padding: 15px;
                border-radius: 8px;
            }
            .notice {
                font-size: 14px;
                color: #555;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <div class="panel">
            <h2>Hospital Appointment No-Show Prediction</h2>
            <p>Enter appointment information available at booking time.</p>

            <textarea id="payload">{{ sample }}</textarea>
            <br>
            <button onclick="predict()">Predict no-show risk</button>

            <h3>Prediction result</h3>
            <pre id="result">Your result will appear here.</pre>

            <p class="notice">
                This tool provides decision support only. A higher-risk
                prediction is not proof that a patient will miss an
                appointment. Staff must review any reminder decision.
                The model must not be used to deny or restrict care.
            </p>
        </div>

        <script>
        async function predict() {
            const output = document.getElementById("result");

            try {
                const payload = JSON.parse(
                    document.getElementById("payload").value
                );

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(payload)
                });

                const result = await response.json();
                output.textContent = JSON.stringify(result, null, 2);

            } catch (error) {
                output.textContent = "Error: " + error.message;
            }
        }
        </script>
    </body>
    </html>
    """, sample=__import__("json").dumps(sample_appointment, indent=4))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


import os
import requests

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:5000")

SAMPLE_INPUT = {
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


def test_health_endpoint():
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200


def test_schema_endpoint():
    response = requests.get(f"{BASE_URL}/schema", timeout=10)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_valid_prediction():
    response = requests.post(
        f"{BASE_URL}/predict",
        json=SAMPLE_INPUT,
        timeout=10
    )
    assert response.status_code == 200

    result = response.json()
    assert 0 <= result["no_show_probability"] <= 1
    assert 0 <= result["attendance_probability"] <= 1
    assert result["human_review_required"] is True


def test_missing_feature_rejected():
    data = SAMPLE_INPUT.copy()
    data.pop("clinic_type")

    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        timeout=10
    )
    assert response.status_code == 400


def test_invalid_numeric_input_rejected():
    data = SAMPLE_INPUT.copy()
    data["lead_time_days"] = "not_a_number"

    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        timeout=10
    )
    assert response.status_code == 400


def test_repeated_prediction_is_consistent():
    first = requests.post(
        f"{BASE_URL}/predict",
        json=SAMPLE_INPUT,
        timeout=10
    ).json()

    second = requests.post(
        f"{BASE_URL}/predict",
        json=SAMPLE_INPUT,
        timeout=10
    ).json()

    assert first["no_show_probability"] == second["no_show_probability"]


def test_invalid_json_rejected():
    response = requests.post(
        f"{BASE_URL}/predict",
        data="{invalid json",
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    assert response.status_code == 400

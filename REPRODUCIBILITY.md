# Reproduction and Testing Guide

## Environment

Python 3.12 is recommended.

## Install dependencies

pip install -r requirements.txt

## Reproduce the project

1. Open the Google Colab notebook.
2. Run all cells in order.
3. Generate the synthetic dataset.
4. Perform preprocessing and feature engineering.
5. Train and compare the classifiers.
6. Evaluate the selected model on the held-out test set.
7. Save the model artifact.
8. Start the Flask API.
9. Run the API tests.

## API endpoints

GET /
GET /health
GET /schema
POST /predict

## Prediction

Send a JSON object containing all 14 required features
to POST /predict.

Use GET /schema to obtain the exact feature list and example input.

## Responsible use

Predictions are for human-reviewed reminder planning only.
They must not be used to deny or restrict healthcare.


## Automated API tests

Start the Flask API by running the relevant notebook cells.
Keep the Colab runtime active, then run:

```bash
pytest -q tests/test_api.py
```

The tests use `http://127.0.0.1:5000` by default.
Set the `BASE_URL` environment variable if your API uses another address.

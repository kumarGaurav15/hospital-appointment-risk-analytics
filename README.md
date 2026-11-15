# Hospital Appointment No-Show Risk Analytics

## 1. Project Overview

This project predicts the probability that a scheduled hospital appointment
will be missed. The purpose is to help staff identify appointments that may
benefit from a reminder, subject to human review.

The project uses synthetic appointment data. It is an operational analytics
prototype, not a clinical decision-making system.

## 2. Problem Definition

- Prediction unit: One scheduled appointment.
- Prediction time: Appointment booking time.
- Target: `no_show`.
- Objective: Estimate the probability of a missed appointment.
- Intended use: Support staff-reviewed reminder planning.

Post-booking information, including `reminder_confirmed` and
`minutes_late_at_checkin`, is excluded from model features to prevent
data leakage.

## 3. Dataset

The synthetic dataset originally contained 6,000 records.
After data cleaning, 5,958 records remained.

| Dataset characteristic | Value |
|---|---:|
| Original records | 6,000 |
| Cleaned records | 5,958 |
| Missed appointments | 1,415 |
| Attended appointments | 4,585 |
| Overall no-show rate | 23.62% |

The dataset is synthetic and was generated using a fixed random seed.
It does not contain real patient records.

### Data split

A chronological 60/20/20 split with a purge step was used.

| Split | Records | No-show rate |
|---|---:|---:|
| Training | 3,411 | 24.2% |
| Validation | 1,026 | 22.1% |
| Test | 1,196 | 23.2% |

The split counts total 5,633. The remaining 325 cleaned records are not
included in these three split counts; refer to the notebook's purge and
boundary outputs for the exact exclusion breakdown.

## 4. Data Preparation and Feature Engineering

The notebook performs:
- Duplicate and invalid-date handling.
- Lead-time validation and recalculation.
- Missing-value imputation.
- Numeric feature scaling.
- Categorical feature encoding.
- Historical appointment feature engineering.
- Exclusion of identifiers, target leakage fields, and post-booking data.

The chronological split and purge step are intended to reduce temporal
information leakage.

## 5. Models and Selection

The following baselines and classifiers were compared using validation
Precision-Recall AUC (PR-AUC).

| Model | Validation PR-AUC |
|---|---:|
| Majority-class baseline | 0.2212 |
| Historical-rule baseline | 0.2552 |
| Random Forest | 0.3170 |
| Gradient Boosting | 0.3372 |
| Logistic Regression | **0.3570** |

Logistic Regression was selected because it achieved the highest validation
PR-AUC among the evaluated models.

Its validation accuracy was 61.89%, and validation recall was 64.32%.

## 6. Probability Calibration and Threshold Selection

Platt Scaling was used to calibrate predicted probabilities.

| Validation Brier score | Value |
|---|---:|
| Before calibration | 0.2315 |
| After calibration | 0.1612 |

The decision threshold was selected on the validation set using the following
relative business-cost assumptions:

- False Negative: $5
- False Positive: $1

The selected threshold was 0.15.

## 7. Final Held-Out Test Results

The selected calibrated Logistic Regression model was evaluated on the
held-out chronological test set.

| Metric | Test result |
|---|---:|
| Recall | **87.00%** |
| Precision | 27.42% |
| F1-score | 41.70% |
| ROC-AUC | 68.70% |
| Brier score | 0.1619 |
| Decision threshold | 0.15 |
| Total business cost | $818 |

### Confusion matrix

| Actual / Predicted | Attended | No-show |
|---|---:|---:|
| Attended | 281 | 638 |
| No-show | 36 | 241 |

The model detected 241 of the 277 actual no-shows and missed 36.
The low threshold prioritises recall, but produces many false positives:
638 attended appointments were flagged as potential no-shows.

This trade-off means predictions should be reviewed by staff before any
reminder action is taken.

## 8. Flask API and Dashboard

The notebook launches a Flask application with these routes:

| Endpoint | Purpose |
|---|---|
| `/` | Staff-facing prediction dashboard |
| `/health` | API health check |
| `/schema` | Required input feature schema |
| `/predict` | Appointment risk prediction |

The prediction endpoint uses the saved model artifact and returns
no-show probability, attendance probability, predicted risk, and a
staff-review reminder suggestion.

The API was tested in the Colab runtime. All eight notebook API checks
were recorded as passed.

## 9. Reproducibility

The notebook contains the data generation, preprocessing, model training,
evaluation, artifact creation, and Flask API implementation.

Install dependencies using:

```bash
pip install -r requirements.txt
```

Run the notebook from top to bottom to regenerate the dataset, train the
model, save the artifact, and launch the API.

With the Flask API running, execute the automated API tests from the
repository root:

```bash
pytest -q tests/test_api.py
```

The test suite expects the API to be running at `http://127.0.0.1:5000`.
Set the `BASE_URL` environment variable if the API is hosted elsewhere.

## 10. Limitations and Responsible Use

- The dataset is synthetic and may not represent real hospital populations.
- Model performance on real-world data may differ.
- The low decision threshold creates a substantial number of false positives.
- Subgroup performance and potential bias should be reviewed before any
  real-world deployment.
- Predictions must not be used to deny, delay, or prioritise clinical care.
- Reminder decisions must remain subject to staff review.
- Real deployment would require privacy, security, governance, and
  prospective validation.

## 11. Project Contents

- `noshow_prediction.ipynb`: Complete notebook workflow.
- `artifacts/noshow_model_artifact.joblib`: Saved trained model artifact.
- `requirements.txt`: Python dependencies.
- `reports/test_report.json`: API test summary.
- `tests/test_api.py`: Reproducible API tests.
- `REPRODUCIBILITY.md`: Instructions for reproducing the project.

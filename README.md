# Electricity Meter Anomaly Detection

**Detecting unusual hourly electricity meter readings with supervised and unsupervised machine learning.**

## Overview

Unexpected electricity meter readings can reflect data errors, equipment faults or unusual usage. This project compares **Isolation Forest** and **LightGBM** on labelled, building-level hourly consumption data. The central challenge is to detect a rare anomaly class while evaluating how well a model transfers to **buildings it has not seen during training**.

## Methods

- Audit timestamps, duplicate records, missing readings, calendar gaps and class imbalance.
- Engineer **24 features**, including seasonal patterns, historical consumption behaviour, consecutive repeated readings and flatline duration.
- Evaluate with **GroupKFold on `building_id`** so individual buildings do not appear in both training and validation.
- Compare manually selected settings with a nested **Gaussian-process Bayesian hyperparameter search**.
- Report PR-AUC, ROC-AUC, precision, recall and F1; choose the classification threshold using out-of-fold predictions.
- Generate test-set predictions and inspect example true-positive anomalies.

## Key results

- **Tuned LightGBM:** mean held-out-fold **PR-AUC ≈ 0.777**, outperforming Isolation Forest in this comparison.
- **Isolation Forest:** tuning improved mean PR-AUC from **0.488 to 0.509**.
- Tuning did **not** materially improve LightGBM over its original hand-picked settings (approximately 0.780 versus 0.777).
- Features detecting repeated or flatlined meter readings were particularly informative.

Flagged observations are candidates for investigation, **not proof of a physical meter fault**.

## Repository contents

- `Question1_Notebook.ipynb` — data processing, modelling, validation and prediction.
- `Question 1 Report.docx` — methodology, results and limitations.
- `Question1_anomaly_submission .csv` — generated anomaly predictions.

## Requirements and reproduction

The **source training and test datasets are not included** in this repository package. To rerun the notebook, obtain authorised copies of `train.csv`, `test.csv` and `sample_submission.csv`.

1. Install the main dependencies:
   ```bash
   pip install numpy pandas scipy matplotlib scikit-learn lightgbm jupyter
   ```
2. In `Question1_Notebook.ipynb`, update `DATA_DIR` (currently `/mnt/user-data/uploads`) to the folder containing those three CSVs.
3. Run all cells in order and review the validation results before regenerating predictions.

## Limitations

The feature design includes retrospectively available information; a live detector would need strictly trailing-window features. Results may differ for new buildings and operating environments. Hyperparameter trials use a training subsample to control computation time.

## Contributors

**Lwazi Nhlapo** and **Fumane Malangabe** — collaborative academic project.

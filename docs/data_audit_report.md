# Data Audit Report

## Primary Dataset

`heart_disease_uci.csv` is used as the primary research dataset because it contains a `dataset` center variable enabling internal-external validation.

- Rows: 920
- Centers: 4
- Binary target absent/present counts: {0: 411, 1: 509}
- Center counts: {'Cleveland': 304, 'Hungary': 293, 'VA Long Beach': 200, 'Switzerland': 123}

## Missingness

Substantial missingness is present, especially in variables such as `ca`, `thal`, and `slope`. Missing values are not imputed globally; imputation is performed inside training folds.

Top missing variables:

| variable   |   missing_count |   missing_percent |
|:-----------|----------------:|------------------:|
| ca         |             611 |         66.413    |
| thal       |             486 |         52.8261   |
| slope      |             309 |         33.587    |
| fbs        |              90 |          9.78261  |
| oldpeak    |              62 |          6.73913  |
| trestbps   |              59 |          6.41304  |
| exang      |              55 |          5.97826  |
| thalach    |              55 |          5.97826  |
| chol       |              30 |          3.26087  |
| restecg    |               2 |          0.217391 |

## Secondary Dataset

`heart.csv` is retained as a supplementary benchmark only.

- Original rows: 1025
- Rows after exact deduplication: 302
- Exact duplicates removed: 723

## Source Manifest

| file                                                                         | sha256                                                           |   rows | role          |
|:-----------------------------------------------------------------------------|:-----------------------------------------------------------------|-------:|:--------------|
| C:\Users\mahmu\OneDrive\Documents\New project\data\raw\heart_disease_uci.csv | 574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae |    920 | primary       |
| C:\Users\mahmu\OneDrive\Documents\New project\data\raw\heart.csv             | ddb2996b2f4db2e00aad13f4518200179ff69f79093838e3c21ffa672ebec0f1 |   1025 | supplementary |

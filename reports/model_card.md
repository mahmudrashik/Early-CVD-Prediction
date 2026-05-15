# Final Model Card

## Model

Selected model: `calibrated_gradient_boosting`

## Intended Use

Educational and research-oriented heart disease risk prediction using public
benchmark data. This is not a medical device and is not intended for clinical
deployment.

## Selection Rationale

The model was selected using a composite ranking of discrimination (AUROC,
AUPRC, MCC, F1), calibration (Brier score, calibration slope), cross-centre
stability (AUROC standard deviation), and interpretability.

## Primary Validation

Leave-one-centre-out internal–external cross-validation using the UCI dataset
centre variable, ensuring geographic generalisability.

## Key Aggregate Metrics

| Metric | Value |
|---|---|
| Mean AUROC | 0.8079 |
| Mean AUPRC | 0.8823 |
| Mean Brier score | 0.1545 |
| Mean Balanced accuracy | 0.7439 |
| Mean F1 | 0.8084 |
| Mean MCC | 0.4390 |
| Mean Sensitivity | 0.8047 |
| Mean Specificity | 0.6832 |

## Bootstrap 95% Confidence Intervals

| metric            |   estimate |   ci_lower |   ci_upper |   bootstrap_iterations |
|:------------------|-----------:|-----------:|-----------:|-----------------------:|
| auroc             |   0.859084 |   0.834852 |   0.883112 |                   2000 |
| auprc             |   0.865913 |   0.831354 |   0.900262 |                   2000 |
| balanced_accuracy |   0.783529 |   0.75738  |   0.809637 |                   2000 |
| brier             |   0.151688 |   0.13887  |   0.164097 |                   2000 |
| sensitivity       |   0.805501 |   0.772462 |   0.840637 |                   2000 |
| specificity       |   0.761557 |   0.721577 |   0.802385 |                   2000 |

## Statistical Significance

DeLong, Friedman–Nemenyi, and McNemar tests are reported in `reports/tables/`.

## Limitations

The public datasets are small, historical, partially missing, and not
representative of modern prospective care. The supplementary `heart.csv` file
has extensive duplicate rows and uncertain target semantics relative to the
UCI `num > 0` definition.

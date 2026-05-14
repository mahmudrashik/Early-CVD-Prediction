# Final Model Card

## Model

Selected model: `logistic_l2`

## Intended Use

Educational and research-oriented heart disease risk prediction using public benchmark data. This is not a medical device and is not intended for clinical deployment.

## Selection Rationale

The model was selected using discrimination, calibration, center-to-center stability, and interpretability. Accuracy alone was not used as the selection criterion.

## Primary Validation

Leave-one-center-out internal-external cross-validation using the UCI dataset center variable.

## Key Aggregate Metrics

- Mean AUROC: 0.799
- Mean AUPRC: 0.880
- Mean Brier score: 0.168
- Mean balanced accuracy: 0.745
- Mean MCC: 0.441

## Limitations

The public datasets are small, historical, partially missing, and not representative of modern prospective care. The supplementary `heart.csv` file has extensive duplicate rows and uncertain target semantics relative to the UCI `num > 0` definition.

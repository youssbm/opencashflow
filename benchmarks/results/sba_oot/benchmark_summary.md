# OpenCashFlow benchmark

| dataset | model | variant | split | n | n_features | default_rate | roc_auc | roc_auc_std | ks | pr_auc | brier | log_loss | fit_seconds | predict_seconds | n_runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sba | lightgbm | all | oot | 20000 | 21 | 0.176 | 0.948 [n/a, n/a] | n/a | 0.768 | 0.866 | 0.101 | 0.318 | 0.791 | 0.068 | 1 |
| sba | logistic | all | oot | 20000 | 21 | 0.176 | 0.784 [n/a, n/a] | n/a | 0.438 | 0.568 | 0.192 | 0.579 | 0.584 | 0.046 | 1 |
| sba | xgboost | all | oot | 20000 | 21 | 0.176 | 0.947 [n/a, n/a] | n/a | 0.780 | 0.848 | 0.106 | 0.322 | 0.798 | 0.037 | 1 |

# OpenCashFlow benchmark

| dataset | model | variant | split | n | n_features | default_rate | roc_auc | roc_auc_std | ks | pr_auc | brier | log_loss | fit_seconds | predict_seconds | n_runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sba | lightgbm | matured-2008 | oot | 20000 | 21 | 0.178 | 0.957 [n/a, n/a] | n/a | 0.797 | 0.891 | 0.091 | 0.290 | 0.923 | 0.074 | 1 |
| sba | logistic | matured-2008 | oot | 20000 | 21 | 0.178 | 0.803 [n/a, n/a] | n/a | 0.469 | 0.650 | 0.192 | 0.574 | 0.762 | 0.041 | 1 |
| sba | xgboost | matured-2008 | oot | 20000 | 21 | 0.178 | 0.957 [n/a, n/a] | n/a | 0.806 | 0.888 | 0.093 | 0.291 | 1.019 | 0.039 | 1 |

# OpenCashFlow benchmark

| dataset | model | variant | split | n | n_features | default_rate | roc_auc | roc_auc_std | ks | pr_auc | brier | log_loss | fit_seconds | predict_seconds | n_runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sba | lightgbm | matured-2008-no-term | oot | 20000 | 19 | 0.178 | 0.639 [n/a, n/a] | n/a | 0.200 | 0.438 | 0.233 | 0.695 | 0.816 | 0.057 | 1 |
| sba | logistic | matured-2008-no-term | oot | 20000 | 19 | 0.178 | 0.581 [n/a, n/a] | n/a | 0.112 | 0.377 | 0.239 | 0.703 | 0.571 | 0.046 | 1 |
| sba | xgboost | matured-2008-no-term | oot | 20000 | 19 | 0.178 | 0.631 [n/a, n/a] | n/a | 0.184 | 0.436 | 0.229 | 0.674 | 0.868 | 0.034 | 1 |

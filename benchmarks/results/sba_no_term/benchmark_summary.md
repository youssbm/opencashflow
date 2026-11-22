# OpenCashFlow benchmark

| dataset | model | variant | split | n | n_features | default_rate | roc_auc | roc_auc_std | ks | pr_auc | brier | log_loss | fit_seconds | predict_seconds | n_runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sba | lightgbm | no-term | stratified | 5000 | 19 | 0.176 | 0.799 [0.784, 0.815] | 0.015 | 0.459 | 0.472 | 0.119 | 0.385 | 0.525 | 0.016 | 6 |
| sba | limix | no-term | stratified | 5000 | 19 | 0.176 | 0.823 [0.808, 0.839] | 0.015 | 0.491 | 0.542 | 0.111 | 0.358 | 0.066 | 66.917 | 6 |
| sba | logistic | no-term | stratified | 5000 | 19 | 0.176 | 0.734 [0.717, 0.751] | 0.016 | 0.365 | 0.363 | 0.130 | 0.415 | 0.205 | 0.015 | 6 |
| sba | xgboost | no-term | stratified | 5000 | 19 | 0.176 | 0.802 [0.788, 0.816] | 0.013 | 0.467 | 0.479 | 0.118 | 0.377 | 0.891 | 0.018 | 6 |

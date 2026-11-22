# OpenCashFlow benchmark

| dataset | model | variant | split | n | n_features | default_rate | roc_auc | roc_auc_std | ks | pr_auc | brier | log_loss | fit_seconds | predict_seconds | n_runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sba | lightgbm | all | stratified | 5000 | 21 | 0.176 | 0.958 [0.948, 0.969] | 0.010 | 0.803 | 0.871 | 0.051 | 0.183 | 0.771 | 0.019 | 6 |
| sba | limix | all | stratified | 5000 | 21 | 0.176 | 0.976 [0.971, 0.981] | 0.004 | 0.867 | 0.913 | 0.039 | 0.142 | 0.061 | 64.157 | 6 |
| sba | logistic | all | stratified | 5000 | 21 | 0.176 | 0.854 [0.843, 0.864] | 0.010 | 0.589 | 0.601 | 0.100 | 0.333 | 0.264 | 0.019 | 6 |
| sba | xgboost | all | stratified | 5000 | 21 | 0.176 | 0.960 [0.950, 0.969] | 0.009 | 0.804 | 0.872 | 0.050 | 0.179 | 0.770 | 0.016 | 6 |

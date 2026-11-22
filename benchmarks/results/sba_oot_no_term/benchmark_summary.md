# OpenCashFlow benchmark

| dataset | model | variant | split | n | n_features | default_rate | roc_auc | roc_auc_std | ks | pr_auc | brier | log_loss | fit_seconds | predict_seconds | n_runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sba | lightgbm | no-term | oot | 20000 | 19 | 0.176 | 0.631 [n/a, n/a] | n/a | 0.186 | 0.392 | 0.216 | 0.668 | 0.740 | 0.061 | 1 |
| sba | logistic | no-term | oot | 20000 | 19 | 0.176 | 0.586 [n/a, n/a] | n/a | 0.132 | 0.331 | 0.226 | 0.686 | 0.367 | 0.037 | 1 |
| sba | xgboost | no-term | oot | 20000 | 19 | 0.176 | 0.620 [n/a, n/a] | n/a | 0.177 | 0.383 | 0.216 | 0.657 | 0.745 | 0.034 | 1 |

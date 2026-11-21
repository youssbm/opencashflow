# OpenCashFlow benchmark

| dataset | model | n | n_features | default_rate | roc_auc | roc_auc_std | ks | brier | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|
| sba | limix | 5000 | 21 | 0.176 | 0.976 | 0.004 | 0.871 | 0.040 | 0.039 |
| sba | xgboost | 5000 | 21 | 0.176 | 0.959 | 0.005 | 0.793 | 0.056 | 0.849 |
| sba | lightgbm | 5000 | 21 | 0.176 | 0.957 | 0.012 | 0.799 | 0.054 | 0.684 |
| sba | logistic | 5000 | 21 | 0.176 | 0.837 | 0.006 | 0.542 | 0.169 | 0.094 |

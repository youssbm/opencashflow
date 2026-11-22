# Raw data directory

This directory is the default landing zone for downloaded benchmark data. **Raw datasets are not
committed to the repository.** Only this README and the `.gitkeep` placeholder are tracked.

## Layout

```
data/
├── download.py
├── README.md
└── raw/
    ├── .gitkeep
    ├── polish.zip                 # downloaded
    ├── polish/1year.arff ...      # extracted
    ├── german.zip                  # downloaded
    ├── german/german.data          # extracted
    ├── bangladesh_sme_1200_records.csv
    └── sba/
        └── SBAnational.csv         # downloaded archive, extracted
```

Set `OPENCASHFLOW_DATA_DIR` to use another directory.

## Download

```bash
python data/download.py --list
python data/download.py --dataset all
python data/download.py --dataset polish german
python data/download.py --dataset bangladesh_sme --force
```

## Licence and attribution

| Dataset | Licence | Citation |
|---|---|---|
| Polish Companies Bankruptcy | CC BY 4.0 | Tomczak, S. (2016). UCI. https://doi.org/10.24432/C5F600 |
| Statlog German Credit | CC BY 4.0 | Hofmann, H. (1994). UCI. https://doi.org/10.24432/C5NC77 |
| Bangladesh SME Credit | CC BY 4.0 data, MIT code | Zenodo record 18525278, version DOI https://doi.org/10.5281/zenodo.18525278 (creators as listed on the Zenodo record) |
| SBA National Loan Data | CC BY-SA 4.0 | Toktogaraev, M. (2020). Kaggle. https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied ; Li, M., Mickel, A. & Taylor, S. (2018). doi 10.1080/10691898.2018.1434342 |

The Bangladesh loader drops `missed_payment_last_12m` because it is an exact restatement of the
target. The SBA loader drops five post-approval or post-default columns from the feature matrix:
`ChgOffDate` and `DisbursementDate` are never loaded for approval-time models, and
`DisbursementGross`, `BalanceGross` and `ChgOffPrinGr` are loaded only with
`include_post_approval=True` for servicing research. See
[docs/datasets.md](../docs/datasets.md) for schema notes and benchmark protocol.

**SBA share-alike:** the Kaggle upload is CC BY-SA 4.0. Do not redistribute the raw CSV, and review
the share-alike obligations before publishing derived feature datasets.

## Do not commit

* raw CSV, ARFF, ZIP or Parquet files;
* bank ledgers or personal data;
* LimiX checkpoints or other model weights;
* credentials, API keys or `.env` files.

The root `.gitignore` already excludes the common cases.

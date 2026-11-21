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
    └── bangladesh_sme_1200_records.csv
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
| Bangladesh SME Credit | CC BY 4.0 data, MIT code | Chowdhury, Das & Paul (2026). Zenodo. https://doi.org/10.5281/zenodo.18525278 |

The Bangladesh loader drops `missed_payment_last_12m` because it is an exact restatement of the
target. See [docs/datasets.md](../docs/datasets.md) for schema notes and benchmark protocol.

## Do not commit

* raw CSV, ARFF, ZIP or Parquet files;
* bank ledgers or personal data;
* LimiX checkpoints or other model weights;
* credentials, API keys or `.env` files.

The root `.gitignore` already excludes the common cases.

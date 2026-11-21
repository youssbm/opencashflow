"""Tests for dataset loaders, attribution and download helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from opencashflow.data.loaders import (
    BANGLADESH_CATEGORICAL,
    DATASET_REGISTRY,
    DatasetBundle,
    _read_arff,
    download_dataset,
    get_data_dir,
    list_datasets,
    load_bangladesh_sme,
    load_german_credit,
    load_polish_bankruptcy,
    load_transactions_csv,
)


def _write_polish(root: Path, rows: int = 2) -> Path:
    directory = root / "polish"
    directory.mkdir(parents=True)
    attributes = "\n".join(f"@attribute Attr{i} numeric" for i in range(1, 65))
    records = []
    for index in range(rows):
        values = [f"{0.1 + index:.2f}"] * 64
        values.append(str(index % 2))
        records.append(",".join(values))
    content = f"@relation test\n{attributes}\n@attribute class {{0,1}}\n@data\n" + "\n".join(
        records
    )
    path = directory / "1year.arff"
    path.write_text(content, encoding="utf-8")
    return path


def test_read_arff(tmp_path):
    path = _write_polish(tmp_path)
    frame = _read_arff(path)
    assert frame.shape == (2, 65)
    assert frame["Attr1"].iloc[0] == pytest.approx(0.1)
    assert frame["class"].tolist() == [0, 1]


def test_polish_loader_from_local_fixture(tmp_path):
    _write_polish(tmp_path)
    bundle = load_polish_bankruptcy(data_dir=tmp_path, download=False)
    assert bundle.X.shape == (2, 64)
    assert bundle.y.tolist() == [0, 1]
    assert bundle.name == "polish"
    assert bundle.metadata["doi"] == "10.24432/C5F600"
    assert "polish" in bundle.summary().lower()
    assert "default rate" in bundle.summary().lower()


def test_polish_all_variants(tmp_path):
    for year in range(1, 6):
        path = tmp_path / "polish" / f"{year}year.arff"
        path.parent.mkdir(parents=True, exist_ok=True)
        attributes = "\n".join(f"@attribute Attr{i} numeric" for i in range(1, 65))
        values = ",".join(["0.5"] * 64) + f",{year % 2}"
        path.write_text(
            f"@relation test\n{attributes}\n@attribute class {{0,1}}\n@data\n{values}\n",
            encoding="utf-8",
        )
    bundle = load_polish_bankruptcy("all", data_dir=tmp_path, download=False)
    assert len(bundle.X) == 5
    assert "horizon_years" in bundle.X.columns
    assert sorted(bundle.X["horizon_years"].unique()) == [1, 2, 3, 4, 5]


def _write_german(root: Path) -> Path:
    directory = root / "german"
    directory.mkdir(parents=True)
    row_good = "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1"
    row_bad = "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2"
    path = directory / "german.data"
    path.write_text(f"{row_good}\n{row_bad}\n", encoding="utf-8")
    return path


def test_german_loader_decodes_categories(tmp_path):
    _write_german(tmp_path)
    bundle = load_german_credit(data_dir=tmp_path, download=False)
    assert bundle.X.shape == (2, 20)
    assert bundle.y.tolist() == [0, 1]
    assert bundle.X.loc[0, "checking_status"] == "< 0 DM"
    assert "checking_status" in bundle.categorical_features
    assert bundle.metadata["cost_matrix"]["approve_bad"] == 5


def test_bangladesh_loader_drops_leakage(tmp_path):
    path = tmp_path / "bangladesh_sme_1200_records.csv"
    records = {
        "business_id": ["B1", "B2"],
        "sector": ["Retail", "Manufacturing"],
        "business_age_years": [5, 12],
        "legal_status": ["Sole", "Limited"],
        "division": ["Dhaka", "Rajshahi"],
        "location_type": ["Urban", "Rural"],
        "employees_count": [4, 20],
        "monthly_revenue_bdt": [100_000, 500_000],
        "monthly_profit_bdt": [10_000, 80_000],
        "existing_loans": ["No", "Yes"],
        "prior_loan_count": [0, 2],
        "late_payment_history": ["Never", "Sometimes"],
        "loan_purpose": ["Working capital", "Equipment"],
        "loan_amount_bdt": [200_000, 1_000_000],
        "preferred_tenure_months": [24, 36],
        "collateral_level": ["Low", "High"],
        "cashflow_stability": [2, 4],
        "missed_payment_last_12m": ["No", "Yes"],
        "repayment_confidence": [2, 5],
        "owner_education": ["Primary", "Master's"],
        "days_payable_outstanding": [45, 20],
        "target_missed_payment": [0, 1],
        "split": ["train", "test"],
    }
    pd.DataFrame(records).to_csv(path, index=False)
    bundle = load_bangladesh_sme(data_dir=tmp_path, download=False)
    assert bundle.X.shape[0] == 2
    assert "missed_payment_last_12m" not in bundle.X.columns
    assert bundle.metadata["dropped_leakage_columns"] == ["missed_payment_last_12m"]
    assert bundle.split.tolist() == ["train", "test"]
    assert set(bundle.categorical_features) <= set(BANGLADESH_CATEGORICAL)
    leaked = load_bangladesh_sme(data_dir=tmp_path, download=False, include_leakage=True)
    assert "missed_payment_last_12m" in leaked.X.columns


def test_transactions_csv_loader(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame(
        {
            "account_id": ["A", "A"],
            "date": ["2024-01-01", "2024-01-02"],
            "amount": [-10.0, 20.0],
            "description": ["RENT", "INVOICE"],
        }
    ).to_csv(path, index=False)
    frame = load_transactions_csv(path)
    assert list(frame["amount"]) == [-10.0, 20.0]
    assert frame["direction"].astype(str).tolist() == ["outflow", "inflow"]


def test_bundle_validation():
    with pytest.raises(ValueError, match="rows"):
        DatasetBundle(name="x", X=pd.DataFrame({"a": [1, 2]}), y=pd.Series([0]))


def test_registry_and_data_dir(tmp_path):
    frame = list_datasets()
    assert set(DATASET_REGISTRY) == {"polish", "german", "bangladesh_sme", "synthetic"}
    assert len(frame) == 4
    assert "license" in frame.columns
    resolved = get_data_dir(tmp_path / "raw")
    assert resolved.is_dir()


def test_download_synthetic_is_local(tmp_path):
    result = download_dataset("synthetic", data_dir=tmp_path)
    assert result == tmp_path


def _real_data_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "raw" / Path(*parts)


@pytest.mark.skipif(
    not _real_data_path("polish", "1year.arff").is_file(),
    reason="Polish dataset not downloaded",
)
def test_real_polish_dataset():
    bundle = load_polish_bankruptcy(download=False)
    assert bundle.X.shape == (7027, 64)
    assert bundle.y.mean() == pytest.approx(0.0386, abs=0.001)


@pytest.mark.skipif(
    not _real_data_path("german", "german.data").is_file(),
    reason="German dataset not downloaded",
)
def test_real_german_dataset():
    bundle = load_german_credit(download=False)
    assert bundle.X.shape == (1000, 20)
    assert bundle.y.mean() == pytest.approx(0.30)


@pytest.mark.skipif(
    not _real_data_path("bangladesh_sme_1200_records.csv").is_file(),
    reason="Bangladesh dataset not downloaded",
)
def test_real_bangladesh_dataset():
    bundle = load_bangladesh_sme(download=False)
    assert bundle.X.shape[0] == 1200
    assert bundle.metadata["dropped_leakage_columns"] == ["missed_payment_last_12m"]

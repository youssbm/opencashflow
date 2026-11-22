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
    load_dataset,
    load_german_credit,
    load_polish_bankruptcy,
    load_sba_loans,
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
    assert bundle.metadata["duplicate_feature_rows"] == 0
    assert "polish" in bundle.summary().lower()
    assert "default rate" in bundle.summary().lower()


def test_polish_duplicate_feature_rows(tmp_path):
    directory = tmp_path / "polish"
    directory.mkdir(parents=True)
    attributes = "\n".join(f"@attribute Attr{i} numeric" for i in range(1, 65))
    first = ",".join(["1.0"] * 64) + ",0"
    second = ",".join(["1.0"] * 64) + ",0"
    other = ",".join(["2.0"] * 64) + ",1"
    (directory / "1year.arff").write_text(
        f"@relation test\n{attributes}\n@attribute class {{0,1}}\n@data\n"
        f"{first}\n{second}\n{other}\n",
        encoding="utf-8",
    )
    bundle = load_polish_bankruptcy(data_dir=tmp_path, download=False)
    assert len(bundle.X) == 3
    assert bundle.metadata["duplicate_feature_rows"] == 2


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


def test_german_exclude_protected(tmp_path):
    _write_german(tmp_path)
    protected = {"personal_status_sex", "age_years", "foreign_worker"}
    full = load_german_credit(data_dir=tmp_path, download=False)
    reduced = load_german_credit(data_dir=tmp_path, download=False, exclude_protected=True)
    assert protected <= set(full.X.columns)
    assert set(full.X.columns) - set(reduced.X.columns) == protected
    assert not protected & set(reduced.categorical_features)
    assert set(reduced.metadata["protected_attributes"]) == protected
    assert reduced.metadata["exclude_protected"] is True


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
    assert bundle.metadata["target_definition"] == (
        "missed any scheduled payment in the 12 months after disbursement (upstream README)"
    )
    assert bundle.metadata["source_branch"] == "zenodo"
    assert "HTTP 403" in bundle.metadata["source_note"]
    assert bundle.split.tolist() == ["train", "test"]
    assert set(bundle.categorical_features) <= set(BANGLADESH_CATEGORICAL)
    leaked = load_bangladesh_sme(data_dir=tmp_path, download=False, include_leakage=True)
    assert "missed_payment_last_12m" in leaked.X.columns


def _write_sba(root: Path) -> Path:
    directory = root / "sba"
    directory.mkdir(parents=True)
    rows = [
        {
            "LoanNr_ChkDgt": "1001",
            "Name": "ALPHA CAFE",
            "City": "AUSTIN",
            "State": "TX",
            "Zip": "78701",
            "Bank": "BANK A",
            "BankState": "TX",
            "NAICS": "722410",
            "ApprovalDate": "28-Feb-97",
            "ApprovalFY": "1997",
            "Term": "84",
            "NoEmp": "4",
            "NewExist": "2",
            "CreateJob": "0",
            "RetainedJob": "4",
            "FranchiseCode": "1",
            "UrbanRural": "1",
            "RevLineCr": "N",
            "LowDoc": "N",
            "ChgOffDate": "",
            "DisbursementDate": "28-Feb-97",
            "DisbursementGross": "$60,000.00",
            "BalanceGross": "$0.00",
            "MIS_Status": "P I F",
            "ChgOffPrinGr": "$0.00",
            "GrAppv": "$60,000.00",
            "SBA_Appv": "$48,000.00",
        },
        {
            "LoanNr_ChkDgt": "1002",
            "Name": "BETA RETAIL",
            "City": "DENVER",
            "State": "CO",
            "Zip": "80202",
            "Bank": "BANK B",
            "BankState": "CO",
            "NAICS": "451120",
            "ApprovalDate": "04-Oct-89",
            "ApprovalFY": "1989",
            "Term": "60",
            "NoEmp": "10",
            "NewExist": "1",
            "CreateJob": "2",
            "RetainedJob": "5",
            "FranchiseCode": "0",
            "UrbanRural": "2",
            "RevLineCr": "Y",
            "LowDoc": "Y",
            "ChgOffDate": "01-Jan-95",
            "DisbursementDate": "04-Oct-89",
            "DisbursementGross": "$100,000.00",
            "BalanceGross": "$25,000.00",
            "MIS_Status": "CHGOFF",
            "ChgOffPrinGr": "$25,000.00",
            "GrAppv": "$100,000.00",
            "SBA_Appv": "$75,000.00",
        },
        {
            "LoanNr_ChkDgt": "1003",
            "Name": "GAMMA WORKS",
            "City": "SEATTLE",
            "State": "WA",
            "Zip": "98101",
            "Bank": "BANK C",
            "BankState": "WA",
            "NAICS": "311812",
            "ApprovalDate": "15-Jun-14",
            "ApprovalFY": "2014",
            "Term": "120",
            "NoEmp": "0",
            "NewExist": "2",
            "CreateJob": "1",
            "RetainedJob": "0",
            "FranchiseCode": "78760",
            "UrbanRural": "0",
            "RevLineCr": "0",
            "LowDoc": "C",
            "ChgOffDate": "",
            "DisbursementDate": "15-Jun-14",
            "DisbursementGross": "$250,000.00",
            "BalanceGross": "$0.00",
            "MIS_Status": "P I F",
            "ChgOffPrinGr": "$0.00",
            "GrAppv": "$250,000.00",
            "SBA_Appv": "$187,500.00",
        },
        {
            "LoanNr_ChkDgt": "1004",
            "Name": "DELTA SERVICES",
            "City": "BOSTON",
            "State": "MA",
            "Zip": "02108",
            "Bank": "BANK D",
            "BankState": "MA",
            "NAICS": "541110",
            "ApprovalDate": "01-Jan-02",
            "ApprovalFY": "2002",
            "Term": "240",
            "NoEmp": "25",
            "NewExist": "1",
            "CreateJob": "0",
            "RetainedJob": "10",
            "FranchiseCode": "1",
            "UrbanRural": "1",
            "RevLineCr": "N",
            "LowDoc": "N",
            "ChgOffDate": "01-Jan-10",
            "DisbursementDate": "01-Jan-02",
            "DisbursementGross": "$500,000.00",
            "BalanceGross": "$100,000.00",
            "MIS_Status": "CHGOFF",
            "ChgOffPrinGr": "$100,000.00",
            "GrAppv": "$500,000.00",
            "SBA_Appv": "$350,000.00",
        },
        {
            "LoanNr_ChkDgt": "1005",
            "Name": "EPSILON UNKNOWN",
            "City": "MIAMI",
            "State": "FL",
            "Zip": "33101",
            "Bank": "BANK E",
            "BankState": "FL",
            "NAICS": "000000",
            "ApprovalDate": "02-Mar-10",
            "ApprovalFY": "2010",
            "Term": "36",
            "NoEmp": "1",
            "NewExist": "1",
            "CreateJob": "0",
            "RetainedJob": "0",
            "FranchiseCode": "1",
            "UrbanRural": "1",
            "RevLineCr": "N",
            "LowDoc": "N",
            "ChgOffDate": "",
            "DisbursementDate": "02-Mar-10",
            "DisbursementGross": "$10,000.00",
            "BalanceGross": "$0.00",
            "MIS_Status": "",
            "ChgOffPrinGr": "$0.00",
            "GrAppv": "$10,000.00",
            "SBA_Appv": "$8,000.00",
        },
    ]
    path = directory / "SBAnational.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_sba_loader_from_local_fixture(tmp_path):
    _write_sba(tmp_path)
    bundle = load_sba_loans(data_dir=tmp_path, download=False)
    assert bundle.X.shape[0] == 4
    assert bundle.y.tolist() == [0, 1, 0, 1]
    assert bundle.metadata["rows_total"] == 4
    assert bundle.metadata["default_rate"] == pytest.approx(0.5)
    assert bundle.metadata["approval_date_min"] == "1989-10-04"
    assert bundle.metadata["approval_date_max"] == "2014-06-15"
    assert bundle.metadata["dropped_leakage_columns"] == [
        "ChgOffDate",
        "DisbursementDate",
        "DisbursementGross",
        "BalanceGross",
        "ChgOffPrinGr",
    ]
    assert bundle.metadata["vintage_min"] is None
    assert bundle.metadata["vintage_max"] is None
    assert "disbursement_gross" not in bundle.X.columns
    assert "loan_id" not in bundle.X.columns
    assert bundle.X.loc[0, "naics_sector"] == "Accommodation and food services"
    assert bundle.X.loc[1, "naics_sector"] == "Retail trade"
    assert bundle.X.loc[2, "naics_sector"] == "Manufacturing"
    assert bundle.X.loc[2, "is_franchise"] == "yes"
    assert bundle.X.loc[0, "sba_guarantee_ratio"] == pytest.approx(0.8)
    assert bundle.X.loc[1, "rev_line_cr"] == "yes"
    assert bundle.X.loc[2, "low_doc"] == "unknown"
    assert str(bundle.X["state"].dtype) == "category"


def test_sba_post_approval_and_identifiers(tmp_path):
    _write_sba(tmp_path)
    bundle = load_sba_loans(
        data_dir=tmp_path,
        download=False,
        include_post_approval=True,
        include_identifiers=True,
    )
    assert {"disbursement_gross", "balance_gross", "chargeoff_amount"} <= set(bundle.X.columns)
    assert {"loan_id", "borrower_name", "city", "zip", "bank"} <= set(bundle.X.columns)
    assert bundle.metadata["post_approval_columns"] == [
        "disbursement_gross",
        "balance_gross",
        "chargeoff_amount",
    ]
    assert len(bundle.metadata["dropped_leakage_columns"]) == 5


def test_sba_vintage_bounds(tmp_path):
    _write_sba(tmp_path)
    upper = load_sba_loans(data_dir=tmp_path, download=False, approval_fy_max=2005)
    assert set(upper.X["approval_fy"]) == {1989, 1997, 2002}
    assert upper.metadata["vintage_min"] is None
    assert upper.metadata["vintage_max"] == 2005
    assert upper.metadata["approval_fy_max"] == 2002
    lower = load_sba_loans(data_dir=tmp_path, download=False, approval_fy_min=2000)
    assert set(lower.X["approval_fy"]) == {2002, 2014}
    assert lower.metadata["vintage_min"] == 2000
    bounded = load_sba_loans(
        data_dir=tmp_path,
        download=False,
        approval_fy_min=1990,
        approval_fy_max=2005,
        max_rows=2,
    )
    assert len(bounded.X) == 2
    assert set(bounded.X["approval_fy"]) <= {1997, 2002}
    with pytest.raises(ValueError, match="approval_fy bounds"):
        load_sba_loans(data_dir=tmp_path, download=False, approval_fy_max=1900)


def test_sba_row_cap_and_dispatch(tmp_path):
    _write_sba(tmp_path)
    capped = load_sba_loans(data_dir=tmp_path, download=False, max_rows=2, random_state=1)
    assert len(capped.X) == 2
    assert set(capped.y.unique()) == {0, 1}
    with pytest.raises(ValueError, match="max_rows"):
        load_sba_loans(data_dir=tmp_path, download=False, max_rows=1)
    dispatched = load_dataset("sba", data_dir=tmp_path, download=False, max_rows=2)
    assert len(dispatched.X) == 2


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
    assert set(DATASET_REGISTRY) == {"polish", "german", "bangladesh_sme", "sba", "synthetic"}
    assert len(frame) == 5
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


@pytest.mark.skipif(
    not _real_data_path("sba", "SBAnational.csv").is_file(),
    reason="SBA dataset not downloaded",
)
def test_real_sba_dataset():
    bundle = load_sba_loans(download=False)
    assert bundle.X.shape == (897_167, 21)
    assert bundle.y.mean() == pytest.approx(0.1756, abs=0.001)
    assert bundle.metadata["rows_total"] == 897_167
    assert "disbursement_gross" not in bundle.X.columns
    assert "ChgOffPrinGr" in bundle.metadata["dropped_leakage_columns"]
    assert bundle.metadata["approval_date_min"] < bundle.metadata["approval_date_max"]

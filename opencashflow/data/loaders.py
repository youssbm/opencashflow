"""Dataset loaders for SME credit-risk benchmarks.

Included datasets
-----------------
``polish``
    UCI Polish Companies Bankruptcy (Tomczak, 2016). 64 financial ratios and a binary bankruptcy
    label, five forecast horizons. CC BY 4.0. DOI 10.24432/C5F600.
``german``
    UCI Statlog German Credit (Hofmann, 1994). 20 coded application attributes and a good/bad
    label. CC BY 4.0. DOI 10.24432/C5NC77. The official cost matrix is 5:1 for approving a bad
    borrower versus rejecting a good one.
``bangladesh_sme``
    1,200 anonymised SME loan applications from Bangladesh (Chowdhury, Das and Paul, 2026).
    Structured features, text narratives and a missed-payment label. CC BY 4.0.
    DOI 10.5281/zenodo.18525278.
``synthetic``
    Generated ledgers from :mod:`opencashflow.data.synthetic`, with a documented distress process.

The UCI and Zenodo files are downloaded on demand into ``data/raw`` (or
``OPENCASHFLOW_DATA_DIR``). Raw data is not redistributed with this repository. Each loader returns
a :class:`DatasetBundle` carrying source, licence and citation metadata so downstream reports can
attribute correctly.
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from opencashflow.data.schema import normalize_transactions

POLISH_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
GERMAN_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
BANGLADESH_ZENODO_URL = (
    "https://zenodo.org/records/18525278/files/shoumyac/bd-sme-research-zenodo.zip?download=1"
)
BANGLADESH_MIRROR_URL = (
    "https://raw.githubusercontent.com/shoumyac/bd-sme-research/zenodo/data/"
    "bangladesh_sme_1200_records.csv"
)

POLISH_DOI = "10.24432/C5F600"
GERMAN_DOI = "10.24432/C5NC77"
BANGLADESH_DOI = "10.5281/zenodo.18525278"

POLISH_VARIANTS = ("1year", "2year", "3year", "4year", "5year")

GERMAN_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_commitment",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property_magnitude",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

GERMAN_CODEBOOK: dict[str, dict[str, str]] = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0-200 DM",
        "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits or all paid",
        "A31": "all credits at this bank paid",
        "A32": "existing credits paid duly",
        "A33": "past delay in payment",
        "A34": "critical account or other",
    },
    "purpose": {
        "A40": "new car",
        "A41": "used car",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "other",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100-500 DM",
        "A63": "500-1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown or none",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1-4 years",
        "A74": "4-7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male divorced",
        "A92": "female divorced or married",
        "A93": "male single",
        "A94": "male married or widowed",
        "A95": "female single",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property_magnitude": {
        "A121": "real estate",
        "A122": "building society or life insurance",
        "A123": "car or other",
        "A124": "unknown or none",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {
        "A171": "unemployed or unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled",
        "A174": "highly qualified",
    },
    "telephone": {"A191": "none", "A192": "yes, registered"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

GERMAN_COST_MATRIX = {"approve_good": 0, "reject_good": 1, "approve_bad": 5, "reject_bad": 0}

BANGLADESH_LEAKAGE_COLUMNS = [
    # This column is a one-to-one restatement of the target; leaving it in produces AUC = 1.0.
    "missed_payment_last_12m",
]
BANGLADESH_TEXT_COLUMNS = [
    "business_description",
    "loan_purpose_description",
    "repayment_plan_description",
    "risk_factors_description",
]
BANGLADESH_CATEGORICAL = [
    "sector",
    "legal_status",
    "division",
    "location_type",
    "existing_loans",
    "late_payment_history",
    "loan_purpose",
    "collateral_level",
    "missed_payment_last_12m",
    "owner_education",
]

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "polish": {
        "title": "Polish Companies Bankruptcy",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data",
        "doi": POLISH_DOI,
        "license": "CC BY 4.0",
        "citation": (
            "Tomczak, S. (2016). Polish Companies Bankruptcy [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5F600"
        ),
        "rows": "7,027 (1year) to 10,503 (3year)",
        "features": 64,
        "default_rate": "3.9%-6.9% depending on horizon",
        "task": "binary bankruptcy classification",
    },
    "german": {
        "title": "Statlog (German Credit Data)",
        "source": "UCI Machine Learning Repository",
        "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        "doi": GERMAN_DOI,
        "license": "CC BY 4.0",
        "citation": (
            "Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77"
        ),
        "rows": "1,000",
        "features": "20 attributes",
        "default_rate": "30.0%",
        "task": "binary credit-risk classification",
    },
    "bangladesh_sme": {
        "title": "Structured Data Suffices: SME Credit Risk in Bangladesh",
        "source": "Zenodo",
        "url": "https://zenodo.org/records/18525278",
        "doi": BANGLADESH_DOI,
        "license": "CC BY 4.0 (data), MIT (code)",
        "citation": (
            "Chowdhury, S., Das, A., & Paul, S. (2026). Structured Data Suffices: Machine Learning "
            "for SME Credit Risk Assessment in an Emerging Market - Dataset and Analysis Code. "
            "Zenodo. https://doi.org/10.5281/zenodo.18525278"
        ),
        "rows": "1,200",
        "features": "27 columns including text narratives",
        "default_rate": "25.25%",
        "task": "binary missed-payment classification",
    },
    "synthetic": {
        "title": "OpenCashFlow synthetic SME portfolio",
        "source": "Generated locally",
        "url": "",
        "doi": "",
        "license": "Apache-2.0 (generated data)",
        "citation": "OpenCashFlow contributors (2026). Synthetic SME transaction generator.",
        "rows": "configurable",
        "features": "151 cash-flow features",
        "default_rate": "configurable, 12% by default",
        "task": "binary default classification",
    },
}


class DatasetNotAvailableError(RuntimeError):
    """Raised when a dataset is not downloaded and automatic download is disabled."""


@dataclass
class DatasetBundle:
    """A benchmark dataset with labels, metadata and attribution.

    Attributes:
        name: Registry key.
        X: Feature matrix.
        y: Binary target, 1 = default / bankruptcy / missed payment.
        target_name: Name of the target column in the source.
        categorical_features: Columns that should be treated as categorical.
        source: Human-readable source.
        license: Data licence.
        citation: Citation string.
        metadata: Extra structured facts.
        split: Optional provided train/test split, aligned to ``X``.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    target_name: str = "default_flag"
    categorical_features: list[str] = field(default_factory=list)
    source: str = ""
    license: str = ""
    citation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    split: pd.Series | None = None

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError(f"X has {len(self.X)} rows but y has {len(self.y)}.")
        self.y = self.y.astype(int).rename(self.target_name)

    @property
    def default_rate(self) -> float:
        """Observed positive rate."""
        return float(self.y.mean()) if len(self.y) else 0.0

    def summary(self) -> str:
        """One-line summary suitable for notebook output."""
        return (
            f"{self.name}: {len(self.X):,} rows x {self.X.shape[1]} features, "
            f"default rate {self.default_rate:.2%}, source {self.source}, licence {self.license}"
        )

    def train_test(
        self, test_size: float = 0.25, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified train/test split, used by examples and notebooks."""
        from sklearn.model_selection import train_test_split

        return train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )


def _project_root() -> Path | None:
    """Find the repository root by looking for pyproject.toml and the package directory."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "opencashflow").is_dir():
            return candidate
    return None


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the raw-data directory, creating it if needed.

    Resolution order: explicit ``data_dir``, ``OPENCASHFLOW_DATA_DIR``, the repository's
    ``data/raw`` when running inside the project, otherwise ``./data/raw``.
    """
    if data_dir is not None:
        path = Path(data_dir).expanduser()
    elif os.getenv("OPENCASHFLOW_DATA_DIR"):
        path = Path(os.environ["OPENCASHFLOW_DATA_DIR"]).expanduser()
    elif (root := _project_root()) is not None:
        path = root / "data" / "raw"
    else:
        path = Path.cwd() / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_file(url: str, destination: Path, *, timeout: int = 120, retries: int = 3) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "OpenCashFlow/0.1 (+https://github.com/opencashflow/opencashflow)",
            "Accept": "*/*",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                destination.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            return destination
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise DatasetNotAvailableError(f"Could not download {url}: {last_error}")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(root)):
                raise DatasetNotAvailableError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(destination)


def download_dataset(
    name: str,
    data_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download and extract one dataset, returning the directory or CSV path.

    Bangladesh SME uses Zenodo as the canonical source and falls back to the authors' GitHub
    mirror when Zenodo is unreachable. The mirror is byte-identical to the CSV inside the archive
    and is documented in ``docs/datasets.md``.
    """
    key = name.lower().strip()
    root = get_data_dir(data_dir)
    if key == "polish":
        target = root / "polish" / "1year.arff"
        if target.is_file() and not force:
            return target
        archive = root / "polish.zip"
        _download_file(POLISH_URL, archive)
        _safe_extract(archive, root / "polish")
        return root / "polish" / "1year.arff"
    if key == "german":
        target = root / "german" / "german.data"
        if target.is_file() and not force:
            return target
        archive = root / "german.zip"
        _download_file(GERMAN_URL, archive)
        _safe_extract(archive, root / "german")
        return root / "german" / "german.data"
    if key in {"bangladesh", "bangladesh_sme"}:
        target = root / "bangladesh_sme_1200_records.csv"
        if target.is_file() and not force:
            return target
        try:
            archive = root / "bd_sme_zenodo.zip"
            _download_file(BANGLADESH_ZENODO_URL, archive)
            _safe_extract(archive, root / "bd_sme_zenodo")
            matches = list((root / "bd_sme_zenodo").rglob("bangladesh_sme_1200_records.csv"))
            if not matches:
                raise DatasetNotAvailableError(
                    "Bangladesh archive did not contain the expected CSV."
                )
            shutil.copyfile(matches[0], target)
        except Exception as exc:
            _download_file(BANGLADESH_MIRROR_URL, target)
            (root / "bangladesh_sme_SOURCE.txt").write_text(
                "Canonical source: https://doi.org/10.5281/zenodo.18525278\n"
                f"Downloaded from mirror because Zenodo was unavailable: {exc}\n",
                encoding="utf-8",
            )
        return target
    if key == "synthetic":
        return root
    raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(DATASET_REGISTRY)}.")


def _read_arff(path: Path) -> pd.DataFrame:
    attributes: list[str] = []
    data_start: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            lower = stripped.lower()
            if lower.startswith("@attribute"):
                parts = stripped.split(maxsplit=2)
                if len(parts) < 2:
                    raise ValueError(f"Malformed @attribute in {path} at line {line_number + 1}")
                attributes.append(parts[1].strip("'\""))
            elif lower.startswith("@data"):
                data_start = line_number + 1
                break
    if data_start is None or not attributes:
        raise ValueError(f"{path} is not a valid ARFF file with attributes and @data.")
    frame = pd.read_csv(
        path,
        skiprows=data_start,
        header=None,
        names=attributes,
        na_values=["?"],
        comment="%",
    )
    return frame


def load_polish_bankruptcy(
    variant: str = "1year",
    data_dir: str | Path | None = None,
    *,
    download: bool = True,
) -> DatasetBundle:
    """Load the UCI Polish Companies Bankruptcy dataset.

    Args:
        variant: ``"1year"`` to ``"5year"``, or ``"all"`` to concatenate all horizons with a
            ``horizon_years`` column. ``1year`` predicts bankruptcy within five years and is the
            primary benchmark.
        data_dir: Raw-data directory. Defaults to ``OPENCASHFLOW_DATA_DIR`` or ``data/raw``.
        download: Download the dataset if missing.

    Returns:
        A :class:`DatasetBundle` with 64 ratio features named ``Attr1`` to ``Attr64``.
    """
    key = variant.lower().replace("_", "").replace("-", "")
    if key != "all" and key not in POLISH_VARIANTS:
        raise ValueError(f"variant must be one of {POLISH_VARIANTS} or 'all'.")
    root = get_data_dir(data_dir)
    if download:
        download_dataset("polish", data_dir=root)
    frames: list[pd.DataFrame] = []
    variants = POLISH_VARIANTS if key == "all" else (key,)
    for current in variants:
        path = root / "polish" / f"{current}.arff"
        if not path.is_file():
            raise DatasetNotAvailableError(
                f"{path} is missing. Call download_dataset('polish') or set download=True."
            )
        frame = _read_arff(path)
        frame["source_file"] = f"{current}.arff"
        frame["horizon_years"] = 6 - int(current[0])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if "class" not in data.columns:
        raise DatasetNotAvailableError("Polish dataset has no 'class' column.")
    y = data["class"].astype(int)
    drop = ["class", "source_file"]
    if key != "all":
        drop.append("horizon_years")
    X = data.drop(columns=drop)
    metadata = dict(DATASET_REGISTRY["polish"])
    metadata.update(
        {
            "variant": key,
            "horizon_years": None if key == "all" else 6 - int(key[0]),
            "positive_rate": float(y.mean()),
            "missing_values": int(X.isna().sum().sum()),
        }
    )
    return DatasetBundle(
        name="polish",
        X=X,
        y=y,
        target_name="bankruptcy_flag",
        categorical_features=[],
        source=metadata["source"],
        license=metadata["license"],
        citation=metadata["citation"],
        metadata=metadata,
    )


def _decode_german(frame: pd.DataFrame) -> pd.DataFrame:
    decoded = frame.copy()
    for column, mapping in GERMAN_CODEBOOK.items():
        if column in decoded.columns:
            decoded[column] = decoded[column].map(mapping).fillna(decoded[column])
    return decoded


def load_german_credit(
    data_dir: str | Path | None = None,
    *,
    download: bool = True,
    numeric: bool = False,
    decode: bool = True,
) -> DatasetBundle:
    """Load the UCI Statlog German Credit dataset.

    Args:
        data_dir: Raw-data directory.
        download: Download the dataset if missing.
        numeric: Use ``german.data-numeric`` (24 numeric attributes) instead of the original
            20-attribute version.
        decode: Map coded categories such as ``A34`` to readable labels.

    Returns:
        A :class:`DatasetBundle` with the positive class being "bad credit risk".
    """
    root = get_data_dir(data_dir)
    if download:
        download_dataset("german", data_dir=root)
    if numeric:
        path = root / "german" / "german.data-numeric"
        if not path.is_file():
            raise DatasetNotAvailableError(f"{path} is missing.")
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        if frame.shape[1] != 25:
            raise DatasetNotAvailableError(
                f"Expected 25 columns in german.data-numeric, found {frame.shape[1]}."
            )
        frame.columns = [f"feature_{i}" for i in range(1, 25)] + ["credit_risk"]
        X = frame.drop(columns="credit_risk")
        y = (frame["credit_risk"] == 2).astype(int)
        categorical: list[str] = []
    else:
        path = root / "german" / "german.data"
        if not path.is_file():
            raise DatasetNotAvailableError(f"{path} is missing.")
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)
        if frame.shape[1] != 21:
            raise DatasetNotAvailableError(
                f"Expected 21 columns in german.data, found {frame.shape[1]}."
            )
        if decode:
            frame = _decode_german(frame)
        X = frame.drop(columns="credit_risk")
        y = (frame["credit_risk"] == 2).astype(int)
        categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    metadata = dict(DATASET_REGISTRY["german"])
    metadata.update(
        {
            "numeric": numeric,
            "decoded": decode,
            "cost_matrix": GERMAN_COST_MATRIX,
            "positive_rate": float(y.mean()),
        }
    )
    return DatasetBundle(
        name="german",
        X=X,
        y=y,
        target_name="bad_credit_flag",
        categorical_features=categorical,
        source=metadata["source"],
        license=metadata["license"],
        citation=metadata["citation"],
        metadata=metadata,
    )


def load_bangladesh_sme(
    data_dir: str | Path | None = None,
    *,
    download: bool = True,
    include_text: bool = False,
    include_leakage: bool = False,
) -> DatasetBundle:
    """Load the 1,200-record Bangladesh SME credit dataset.

    Args:
        data_dir: Raw-data directory.
        download: Download the dataset if missing.
        include_text: Keep the four narrative columns. They are useful for NLP experiments but
            should be excluded from tabular baselines to avoid high-cardinality leakage.
        include_leakage: Keep ``missed_payment_last_12m``. This column is a one-to-one restatement
            of the target, so it is dropped by default. Enable only to reproduce the leaky version
            for teaching purposes.

    Returns:
        A :class:`DatasetBundle` with a missed-payment label and the provided train/test split.
    """
    root = get_data_dir(data_dir)
    if download:
        download_dataset("bangladesh_sme", data_dir=root)
    path = root / "bangladesh_sme_1200_records.csv"
    if not path.is_file():
        raise DatasetNotAvailableError(
            f"{path} is missing. Call download_dataset('bangladesh_sme')."
        )
    frame = pd.read_csv(path)
    target = "target_missed_payment"
    if target not in frame.columns:
        raise DatasetNotAvailableError(f"Expected target column {target!r} in {path}.")
    y = frame[target].astype(int)
    drop = [target]
    if "business_id" in frame.columns:
        drop.append("business_id")
    split = frame["split"].copy() if "split" in frame.columns else None
    if "split" in frame.columns:
        drop.append("split")
    if not include_text:
        drop.extend([c for c in BANGLADESH_TEXT_COLUMNS if c in frame.columns])
    dropped_leakage: list[str] = []
    if not include_leakage:
        dropped_leakage = [c for c in BANGLADESH_LEAKAGE_COLUMNS if c in frame.columns]
        drop.extend(dropped_leakage)
    X = frame.drop(columns=drop)
    categorical = [c for c in BANGLADESH_CATEGORICAL if c in X.columns]
    for column in categorical:
        X[column] = X[column].astype("category")
    metadata = dict(DATASET_REGISTRY["bangladesh_sme"])
    metadata.update(
        {
            "include_text": include_text,
            "include_leakage": include_leakage,
            "dropped_leakage_columns": dropped_leakage,
            "positive_rate": float(y.mean()),
            "provided_split": split.value_counts().to_dict() if split is not None else None,
        }
    )
    return DatasetBundle(
        name="bangladesh_sme",
        X=X,
        y=y,
        target_name="missed_payment_flag",
        categorical_features=categorical,
        source=metadata["source"],
        license=metadata["license"],
        citation=metadata["citation"],
        metadata=metadata,
        split=split,
    )


def load_synthetic(
    n_companies: int = 120,
    months: int = 24,
    default_rate: float = 0.12,
    seed: int = 42,
    *,
    window_days: int | None = None,
    as_bundle: bool = True,
) -> DatasetBundle | tuple[pd.DataFrame, pd.Series]:
    """Generate the synthetic portfolio and return a bundle (or ``X`` and ``y``).

    Feature extraction uses ``window_days=months * 31`` by default so the full generated history is
    used. Pass ``window_days`` to simulate a shorter observation window.
    """
    from opencashflow.data.synthetic import generate_credit_dataset

    X, y, portfolio = generate_credit_dataset(
        n_companies=n_companies,
        months=months,
        default_rate=default_rate,
        seed=seed,
        window_days=window_days,
    )
    if not as_bundle:
        return X, y
    metadata = dict(DATASET_REGISTRY["synthetic"])
    metadata.update(portfolio.metadata)
    return DatasetBundle(
        name="synthetic",
        X=X,
        y=y,
        target_name="default_flag",
        categorical_features=[],
        source=metadata["source"],
        license=metadata["license"],
        citation=metadata["citation"],
        metadata=metadata,
    )


def load_transactions_csv(
    path: str | Path,
    *,
    mapping: dict[str, str] | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a transaction CSV and normalise it to the canonical schema.

    Extra keyword arguments are forwarded to :func:`pandas.read_csv`.
    """
    frame = pd.read_csv(path, **read_csv_kwargs)
    return normalize_transactions(frame, mapping=mapping)


def load_dataset(name: str, **kwargs: Any) -> DatasetBundle:
    """Load a dataset by registry name.

    Names: ``polish``, ``german``, ``bangladesh_sme`` (or ``bangladesh``), ``synthetic``.
    """
    key = name.lower().strip().replace("-", "_")
    if key in {"polish", "polish_bankruptcy"}:
        return load_polish_bankruptcy(**kwargs)
    if key in {"german", "german_credit"}:
        return load_german_credit(**kwargs)
    if key in {"bangladesh", "bangladesh_sme"}:
        return load_bangladesh_sme(**kwargs)
    if key in {"synthetic", "sme_synthetic"}:
        return load_synthetic(**kwargs)
    raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(DATASET_REGISTRY)}.")


def list_datasets() -> pd.DataFrame:
    """Return the dataset registry as a DataFrame."""
    return pd.DataFrame(
        [
            {
                "name": key,
                "title": value["title"],
                "rows": value["rows"],
                "features": value["features"],
                "default_rate": value["default_rate"],
                "license": value["license"],
                "doi": value["doi"],
            }
            for key, value in DATASET_REGISTRY.items()
        ]
    )


def describe_datasets() -> str:
    """Return a human-readable dataset registry table."""
    return list_datasets().to_string(index=False)


__all__ = [
    "DATASET_REGISTRY",
    "DatasetBundle",
    "DatasetNotAvailableError",
    "describe_datasets",
    "download_dataset",
    "get_data_dir",
    "list_datasets",
    "load_bangladesh_sme",
    "load_dataset",
    "load_german_credit",
    "load_polish_bankruptcy",
    "load_synthetic",
    "load_transactions_csv",
]

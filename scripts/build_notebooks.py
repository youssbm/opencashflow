"""Build the three example notebooks.

The notebooks are generated from a script so their structure stays reproducible and easy to review.
Run this after changing the public API, then execute them with nbconvert.

Usage::

    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_feature_engineering.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(source) if kind == "markdown" else nbf.v4.new_code_cell(source)
        for kind, source in cells
    ]
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return notebook


def build_feature_notebook() -> nbf.NotebookNode:
    cells = [
        (
            "markdown",
            """# 01 - Feature engineering

This notebook turns a synthetic SME transaction ledger into the 151-feature credit matrix.

You will see:

1. the canonical transaction schema;
2. rule-based categorisation of localised bank narratives;
3. monthly cash-flow behaviour for one company;
4. the full feature matrix, group counts and distributions;
5. how a few features separate defaulters from non-defaulters.

The data is generated locally, so the notebook runs without downloads or credentials.
""",
        ),
        (
            "code",
            """import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

from opencashflow.data.synthetic import generate_portfolio
from opencashflow.features.categorizer import EXPENSE_CATEGORIES, TransactionCategorizer
from opencashflow.features.cashflow_features import CashFlowFeatureExtractor, FeatureConfig
from opencashflow.features.specs import feature_group_counts

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
plt.style.use("default")""",
        ),
        (
            "markdown",
            "## 1. Generate a portfolio\n\nSixty companies, 18 months of history, six European countries and a 15% target default rate.",
        ),
        (
            "code",
            """portfolio = generate_portfolio(
    n_companies=60,
    months=18,
    default_rate=0.15,
    seed=42,
)
print(portfolio.summary())
portfolio.transactions.head()""",
        ),
        (
            "markdown",
            "## 2. Categorise one company's ledger\n\nThe categoriser is rule-first and direction-aware. Every prediction records the category, confidence, method and matched rule.",
        ),
        (
            "code",
            """example_id = portfolio.transactions["entity_id"].iloc[0]
ledger = portfolio.transactions[portfolio.transactions["entity_id"] == example_id].copy()
categorizer = TransactionCategorizer()
labelled = categorizer.categorize_frame(ledger)

print(f"Entity: {example_id}")
print("Uncategorised share:", round(labelled["category_method"].eq("fallback").mean(), 3))
categorizer.category_summary(labelled)""",
        ),
        (
            "code",
            """summary = categorizer.category_summary(labelled).head(10)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(summary["category"], summary["abs_amount"], color="#3b6ea5")
ax.invert_yaxis()
ax.set_xlabel("Absolute flow (EUR)")
ax.set_title(f"Largest transaction categories: {example_id}")
plt.tight_layout()
plt.show()""",
        ),
        (
            "markdown",
            "## 3. Monthly cash-flow behaviour\n\nRevenue is customer receipts only. Operating expenses exclude loan principal, which is tracked as debt service.",
        ),
        (
            "code",
            """labelled["month"] = labelled["date"].dt.to_period("M").astype(str)
expense_mask = labelled["category"].isin(EXPENSE_CATEGORIES) & (labelled["amount"] < 0)
revenue_mask = (labelled["category"] == "revenue") & (labelled["amount"] > 0)

monthly = pd.DataFrame(
    {
        "inflow": labelled.loc[labelled["amount"] > 0].groupby("month")["amount"].sum(),
        "outflow": -labelled.loc[labelled["amount"] < 0].groupby("month")["amount"].sum(),
        "revenue": labelled.loc[revenue_mask].groupby("month")["amount"].sum(),
        "expenses": -labelled.loc[expense_mask].groupby("month")["amount"].sum(),
    }
).fillna(0.0)

fig, ax = plt.subplots(figsize=(10, 4))
monthly[["revenue", "expenses"]].plot(ax=ax, marker="o")
ax.set_title(f"Monthly revenue and operating expenses: {example_id}")
ax.set_xlabel("Month")
ax.set_ylabel("EUR")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()""",
        ),
        (
            "markdown",
            "## 4. Build the full feature matrix\n\nOne row per borrower, 151 columns. The extractor uses a two-year window here so the full generated history is used.",
        ),
        (
            "code",
            """extractor = CashFlowFeatureExtractor(config=FeatureConfig(window_days=18 * 31))
X = extractor.fit_transform(portfolio.transactions)
y = portfolio.labels.set_index("entity_id")["default_flag"].reindex(X.index)

print("Feature matrix:", X.shape)
print("Default rate:", round(float(y.mean()), 3))
print("All finite:", bool(X.notna().all().all()))
X[["revenue_total", "revenue_cv", "cash_buffer_months", "nsf_rate", "cashflow_health_score"]].describe().T""",
        ),
        (
            "code",
            """counts = pd.Series(feature_group_counts()).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4))
counts.plot(kind="barh", ax=ax, color="#5b8c5a")
ax.set_xlabel("Number of features")
ax.set_title("Feature catalogue by group (151 total)")
plt.tight_layout()
plt.show()""",
        ),
        (
            "markdown",
            "## 5. Do the features separate defaults?\n\nThe synthetic labels are generated from post-ledger outcomes such as low balances, negative cash-flow months, failed payments and falling inflows. The features should recover that signal without seeing the latent stress variable.",
        ),
        (
            "code",
            """fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(X.loc[y == 0, "cashflow_health_score"], bins=15, alpha=0.65, label="No default", color="#3b6ea5")
ax.hist(X.loc[y == 1, "cashflow_health_score"], bins=15, alpha=0.65, label="Default", color="#c44e52")
ax.set_xlabel("Cash-flow health score (0-100)")
ax.set_ylabel("Companies")
ax.set_title("Health score distribution by outcome")
ax.legend()
plt.tight_layout()
plt.show()

pd.DataFrame(
    {
        "mean_no_default": X.loc[y == 0, ["revenue_cv", "net_cash_flow_negative_ratio", "overdraft_ratio", "cashflow_health_score"]].mean(),
        "mean_default": X.loc[y == 1, ["revenue_cv", "net_cash_flow_negative_ratio", "overdraft_ratio", "cashflow_health_score"]].mean(),
    }
).round(3)""",
        ),
        (
            "code",
            """selected = [
    "revenue_cv",
    "revenue_growth_3m",
    "net_cash_flow_negative_ratio",
    "overdraft_ratio",
    "dscr",
    "nsf_rate",
    "cash_buffer_months",
    "cashflow_health_score",
]
corr = X[selected].corr()

fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(selected)), selected, rotation=45, ha="right")
ax.set_yticks(range(len(selected)), selected)
ax.set_title("Selected feature correlations")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()""",
        ),
        (
            "markdown",
            "## 6. The feature dictionary\n\nEvery feature has a group, description, direction and formula. The catalogue is the single source of truth used by the extractor, docs and tests.",
        ),
        (
            "code",
            """extractor.describe_features().query("group == 'liquidity'").head(10)[
    ["name", "description", "higher_is_better"]
]""",
        ),
        (
            "markdown",
            """## Takeaways

* The categoriser turns localised narratives into consistent credit categories with an audit trail.
* The feature matrix is wide but documented, and every value is finite.
* Balance, negative-month, overdraft and health-score features separate outcomes in the synthetic benchmark.
* For production, pass a fixed `as_of` date and validate on out-of-time data.

Next: [02_limix_scoring.ipynb](02_limix_scoring.ipynb) for PD modelling and explanations.
""",
        ),
    ]
    return _notebook(cells)


def build_limix_notebook() -> nbf.NotebookNode:
    cells = [
        (
            "markdown",
            """# 02 - LimiX-2 scoring and explanations

This notebook fits LimiX-2, LightGBM and logistic regression on the same synthetic cash-flow
portfolio, compares ranking and calibration, and explains individual decisions.

> **Licence:** LimiX-2 weights are non-commercial. The notebook detects whether the checkpoint is
> available and falls back to the tree and linear baselines otherwise. See `docs/limix.md`.
""",
        ),
        (
            "code",
            """import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from opencashflow.data.synthetic import generate_portfolio
from opencashflow.features.cashflow_features import FeatureConfig
from opencashflow.models import (
    LimiXScorer,
    LogisticBaseline,
    TreeBaseline,
    calibration_table,
    evaluate_binary,
    explain_model,
)

pd.set_option("display.width", 140)""",
        ),
        (
            "markdown",
            "## 1. Build a labelled portfolio\n\nWe use 120 companies and a two-year feature window. The train/test split is stratified so the default rate is similar in both sets.",
        ),
        (
            "code",
            """portfolio = generate_portfolio(
    n_companies=120,
    months=18,
    default_rate=0.15,
    seed=7,
)
X, y = portfolio.to_features(FeatureConfig(window_days=18 * 31))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(portfolio.summary())
print("Train:", X_train.shape, "Test:", X_test.shape, "Default rate:", round(float(y.mean()), 3))""",
        ),
        (
            "markdown",
            "## 2. Check LimiX availability\n\nThe wrapper resolves the checkpoint from `LIMIX_MODEL_PATH` and the configs from `LIMIX_SRC` / `LIMIX_CONFIG_DIR`. No weights are downloaded automatically.",
        ),
        (
            "code",
            """limix_info = LimiXScorer.availability()
limix_info""",
        ),
        (
            "markdown",
            "## 3. Fit the models\n\nLimiX uses in-context learning: `fit` stores the labelled reference set and `predict` runs the frozen checkpoint. One pipeline is used on CPU.",
        ),
        (
            "code",
            """models = {
    "Logistic": LogisticBaseline(),
    "LightGBM": TreeBaseline("lightgbm", n_estimators=300),
}
if limix_info["available"]:
    models["LimiX-2 (1 pipeline)"] = LimiXScorer(n_pipelines=1, seed=0, verbose=False)
else:
    print("LimiX unavailable; using tree and linear baselines only.")

scores = {}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    probabilities = model.predict_pd(X_test)
    scores[name] = probabilities
    metrics = evaluate_binary(y_test, probabilities)
    rows.append(
        {
            "model": name,
            "roc_auc": metrics["roc_auc"],
            "ks": metrics["ks"],
            "brier": metrics["brier"],
            "log_loss": metrics["log_loss"],
        }
    )

metrics_frame = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
metrics_frame.round(3)""",
        ),
        (
            "markdown",
            "## 4. Ranking and calibration\n\nAUC measures ranking. Brier score and log loss also penalise overconfident probabilities, which matters for pricing and provisioning.",
        ),
        (
            "code",
            """fig, ax = plt.subplots(figsize=(7, 5))
for name, probabilities in scores.items():
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probabilities):.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the synthetic holdout")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """best_name = metrics_frame.iloc[0]["model"]
best_probabilities = scores[best_name]
calibration = calibration_table(y_test, best_probabilities, n_bins=8)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect calibration")
ax.plot(calibration["mean_predicted"], calibration["observed"], marker="o", label=best_name)
ax.set_xlabel("Mean predicted PD")
ax.set_ylabel("Observed default rate")
ax.set_title(f"Calibration: {best_name}")
ax.legend()
plt.tight_layout()
plt.show()

calibration.round(3)""",
        ),
        (
            "markdown",
            "## 5. Explain the riskiest application\n\nTree models use exact TreeSHAP. LimiX uses occlusion: replace one feature at a time with a reference value and measure the PD change. Reason codes are model-relative, not causal.",
        ),
        (
            "code",
            """best_model = models[best_name]
order = np.argsort(-best_probabilities)[:3]
selected = X_test.iloc[order]
explanations = explain_model(best_model, selected, max_features=25)

for explanation in explanations:
    print(f"\\nEntity {explanation.entity_id} | PD={explanation.prediction:.3f} | method={explanation.method}")
    for reason in explanation.reason_codes(4):
        print("  -", reason)""",
        ),
        (
            "code",
            """explanation = explanations[0]
explanation.top(8)[["feature", "value", "contribution", "direction", "description"]]""",
        ),
        (
            "markdown",
            "## 6. What to remember\n\n* LimiX is competitive on small tabular samples but costs more latency than trees.\n* A high AUC can hide poor calibration; report Brier and log loss.\n* Explanations are decision support, not a causal claim or a complete adverse-action process.\n* One CPU pipeline is a fast demo, not a leaderboard reproduction.",
        ),
        (
            "code",
            """for model in models.values():
    if hasattr(model, "close"):
        model.close()
print("Released LimiX predictor memory.")""",
        ),
    ]
    return _notebook(cells)


def build_benchmark_notebook() -> nbf.NotebookNode:
    cells = [
        (
            "markdown",
            """# 03 - Benchmarks

Compares LimiX-2, LightGBM, XGBoost, logistic regression and the heuristic scorecard on the public
and synthetic datasets with stratified cross-validation.

The protocol is deliberately conservative:

* the same folds for every model;
* preprocessing fitted inside each training fold;
* LimiX receives only the training fold as its in-context reference set;
* AUC, KS, PR-AUC, Brier and log loss are reported.

The Polish benchmark is omitted here for runtime; run the CLI for all four datasets.
""",
        ),
        (
            "code",
            """import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from opencashflow.benchmark import run_benchmark
from opencashflow.data.loaders import list_datasets

pd.set_option("display.width", 160)""",
        ),
        (
            "markdown",
            "## 1. Dataset registry",
        ),
        (
            "code",
            """list_datasets()""",
        ),
        (
            "markdown",
            "## 2. Run the benchmark\n\nThis uses three datasets, three folds, and one LimiX pipeline. Expect a couple of minutes on CPU. If LimiX is unavailable, the harness skips it with a warning.",
        ),
        (
            "code",
            """root = Path.cwd()
if not (root / "pyproject.toml").exists() and (root.parent / "pyproject.toml").exists():
    root = root.parent

report = run_benchmark(
    datasets=["synthetic", "german", "bangladesh_sme"],
    models=["heuristic", "logistic", "lightgbm", "xgboost", "limix"],
    cv=3,
    seed=42,
    synthetic_companies=160,
    synthetic_months=18,
    limix_pipelines=1,
    output_dir=root / "benchmarks" / "results" / "notebook",
    verbose=True,
)
report.summary.round(3)""",
        ),
        (
            "markdown",
            "## 3. Ranking by dataset",
        ),
        (
            "code",
            """pivot = report.summary.pivot(index="dataset", columns="model", values="roc_auc").sort_index()
fig, ax = plt.subplots(figsize=(11, 4))
pivot.plot(kind="bar", ax=ax)
ax.set_ylabel("ROC-AUC (3-fold mean)")
ax.set_title("Model comparison by dataset")
ax.set_ylim(0.5, 1.0)
ax.legend(loc="lower right", fontsize=8)
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()
pivot.round(3)""",
        ),
        (
            "markdown",
            "## 4. Calibration matters too\n\nAUC only measures ranking. Brier score penalises confident wrong probabilities.",
        ),
        (
            "code",
            """brier = report.summary.pivot(index="dataset", columns="model", values="brier").sort_index()
brier.round(3)""",
        ),
        (
            "markdown",
            "## 5. Dataset notes\n\nThe Bangladesh loader drops `missed_payment_last_12m` because it is an exact copy of the target. German and Polish are tabular accounting/application datasets, not transaction ledgers.",
        ),
        (
            "code",
            """from opencashflow.data.loaders import load_dataset

for name in ["german", "bangladesh_sme"]:
    bundle = load_dataset(name)
    print(bundle.summary())
    print("  leakage columns dropped:", bundle.metadata.get("dropped_leakage_columns", []))
    print("  categorical features:", len(bundle.categorical_features))""",
        ),
        (
            "markdown",
            """## 6. Interpretation guardrails

* These are small public datasets. Fold variance is real; read `roc_auc_std` in the results table.
* The synthetic portfolio is the only transaction-level benchmark and its labels are generated.
* LimiX ran with one pipeline on CPU. A 32-member GPU run is not comparable.
* Do not claim state of the art from this notebook. Use it to compare implementations, features and
  preprocessing under a shared protocol.

Saved outputs are in `benchmarks/results/notebook/`.
""",
        ),
    ]
    return _notebook(cells)


def main() -> int:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "01_feature_engineering.ipynb": build_feature_notebook(),
        "02_limix_scoring.ipynb": build_limix_notebook(),
        "03_benchmarks.ipynb": build_benchmark_notebook(),
    }
    for name, notebook in notebooks.items():
        path = NOTEBOOKS / name
        nbf.write(notebook, path)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

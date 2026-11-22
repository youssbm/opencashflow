"""LimiX-2 in-context scorer.

LimiX-2 is a tabular foundation model from Stable AI. One checkpoint handles classification,
regression and missing-value imputation; there is no gradient training in this wrapper. ``fit``
stores the labelled reference set and ``predict`` performs in-context inference.

Licence
-------
The LimiX-2 weights are released under the **StableAI LimiX Non-Commercial License v1.0**.
Commercial use requires a separate agreement. If you publish results, disclose the model name
"LimiX" as required by the licence. This wrapper is Apache-2.0; the licence of the weights is
separate and stricter.

Local setup
-----------
1. Clone the code at commit ``a917fa0`` and install it::

       git clone https://github.com/limix-ldm-ai/LimiX
       cd LimiX && git checkout a917fa0 && pip install -e .

2. Download ``LimiX-2-1016.ckpt`` from https://huggingface.co/stable-ai/LimiX-2-1016
   (commit ``bbbf26c``) and point ``LIMIX_MODEL_PATH`` at it.

3. Set ``LIMIX_CONFIG_DIR`` to the repository's ``config`` directory if it is not discoverable.

On CPU-only Windows, a no-op ``triton`` shim is installed automatically before importing LimiX,
because LimiX imports triton even when its default path is pure PyTorch.

Temperature
-----------
``softmax_temperature`` defaults to ``None``. It resolves to 0.9 for classification, which keeps
benchmark runs comparable, and to 1.0 for regression, which is required for leaderboard-equivalent
reproduction. Pass a float to override either value. The resolved value is stored on the instance
and reported by :meth:`describe`.

Categorical columns
-------------------
Object, string, boolean and pandas categorical columns are ordinal-encoded with codes learned at
``fit``. Categories are sorted alphabetically, so the resulting integer codes are arbitrary labels
without an ordering meaning. The wrapper does not pass categorical indices to LimiX. Upstream
LimiX infers them itself from the training rows: a column is treated as categorical when it has
fewer than ``min_unique_num_for_numerical_infer = 4`` unique training values, and only when the
training set has at least ``min_seq_len_for_category_infer = 100`` rows. Categories unseen at
``fit`` encode to -1. With small training sets neither inference branch triggers, so the ordinal
codes are handled as plain numbers.

Known limitation
----------------
Classification median-fills missing values before the matrix reaches LimiX, so the information
carried by the pattern of missingness is lost. TreeBaseline, by contrast, keeps NaN and can use
that signal. On the Polish bankruptcy data the NaN count alone reaches an AUC of 0.725. This is a
documented limitation of the classification path; imputation is handled separately by
:meth:`impute`, which passes NaN through to LimiX's feature-imputation task.

Example:
    >>> from opencashflow.models.limix_scorer import LimiXScorer
    >>> scorer = LimiXScorer(n_pipelines=1)  # 1 pipeline is fast on CPU
    >>> scorer.fit(X_train, y_train)
    >>> pd_values = scorer.predict_pd(X_test)
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

LICENSE_NOTICE = (
    "LimiX-2 weights are licensed under the StableAI LimiX Non-Commercial License v1.0. "
    "Commercial use requires a separate agreement. Cite the model as 'LimiX' when publishing."
)

DEFAULT_CHECKPOINT = "LimiX-2-1016.ckpt"
CLASSIFICATION_CONFIG = "cls_default_noretrieval_v2.json"
REGRESSION_CONFIG = "reg_default_noretrieval_v2.json"
IMPUTATION_CONFIG = "reg_default_noretrieval_MVI_v2.json"


def _is_categorical(series: pd.Series) -> bool:
    """Whether a pandas column should be ordinal-encoded before entering LimiX."""
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def _make_dataset_name(seed: int, X: np.ndarray, y: Any) -> str:
    """Return a short, content-addressed LimiX dataset name.

    The digest covers both the feature matrix and the labels so two different training sets with
    the same seed cannot share a preprocessing cache entry. The helper is pure and needs no LimiX
    import, so it can be unit tested on any machine.

    Args:
        seed: Random seed, kept in the name for continuity with earlier runs.
        X: Feature matrix actually passed to LimiX.
        y: Labels actually passed to LimiX; numeric and string labels are both accepted.

    Returns:
        A name of the form ``opencashflow-{seed}-{16 hex characters}``.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    digest.update(repr(np.asarray(y).tolist()).encode("utf-8"))
    return f"opencashflow-{seed}-{digest.hexdigest()[:16]}"


class LimiXUnavailableError(RuntimeError):
    """Raised when the LimiX code, checkpoint or configuration cannot be found."""


def _install_optional_shims() -> None:
    """Install no-op triton and nvtx shims when the real packages are unavailable.

    LimiX imports triton unconditionally but uses pure PyTorch kernels by default. On CPU-only
    Windows this shim is required. ``nvtx`` is only used for profiling annotations.
    """
    if "triton" not in sys.modules and importlib.util.find_spec("triton") is None:

        def _jit(fn: Any = None, **kwargs: Any) -> Any:
            del kwargs
            return fn if fn is not None else (lambda f: f)

        triton_language = types.ModuleType("triton.language")
        triton_language.constexpr = int  # type: ignore[attr-defined]
        triton = types.ModuleType("triton")
        triton.jit = _jit  # type: ignore[attr-defined]
        triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()  # type: ignore[attr-defined]
        triton.language = triton_language  # type: ignore[attr-defined]
        sys.modules.setdefault("triton", triton)
        sys.modules.setdefault("triton.language", triton_language)

    if "nvtx" not in sys.modules and importlib.util.find_spec("nvtx") is None:

        class _Annotation:
            def __enter__(self) -> _Annotation:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        nvtx = types.ModuleType("nvtx")
        nvtx.annotate = lambda *args, **kwargs: _Annotation()  # type: ignore[attr-defined]
        sys.modules.setdefault("nvtx", nvtx)


def _search_checkpoint() -> Path | None:
    env = os.getenv("LIMIX_MODEL_PATH") or os.getenv("OPENCASHFLOW_LIMIX_CKPT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            Path.cwd() / "cache_limix" / DEFAULT_CHECKPOINT,
            Path.cwd() / "models" / DEFAULT_CHECKPOINT,
            Path.home() / ".cache" / "opencashflow" / "limix" / DEFAULT_CHECKPOINT,
            Path.home() / ".cache" / "limix" / DEFAULT_CHECKPOINT,
        ]
    )
    hf_root = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_root.exists():
        for pattern in (
            "models--stable-ai--LimiX-2-1016/snapshots/*/*.ckpt",
            "models--stable-ai--LimiX-2*/snapshots/*/*.ckpt",
        ):
            candidates.extend(hf_root.glob(pattern))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _search_config_dir() -> Path | None:
    env = os.getenv("LIMIX_CONFIG_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    src = os.getenv("LIMIX_SRC")
    if src:
        candidates.append(Path(src).expanduser() / "config")
    candidates.append(Path.cwd() / "limix_src" / "config")
    try:
        import limix

        candidates.append(Path(limix.__file__).resolve().parent.parent / "config")
    except ImportError:
        pass
    for candidate in candidates:
        if (candidate / CLASSIFICATION_CONFIG).is_file():
            return candidate
    return None


def _resolve_limix_src() -> Path | None:
    env = os.getenv("LIMIX_SRC")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.cwd() / "limix_src")
    try:
        import limix

        candidates.append(Path(limix.__file__).resolve().parent.parent)
    except ImportError:
        pass
    for candidate in candidates:
        if (candidate / "inference" / "predictor.py").is_file():
            return candidate
    return None


def _load_predictor_class() -> type:
    _install_optional_shims()
    try:
        from inference.predictor import LimiXPredictor  # type: ignore

        return LimiXPredictor
    except Exception as first_error:
        src = _resolve_limix_src()
        if src is None:
            raise LimiXUnavailableError(
                "Could not import LimiX. Install the LimiX repository and set LIMIX_SRC to its "
                "root (the folder containing 'inference/'), and LIMIX_MODEL_PATH to the "
                f"checkpoint. Original error: {first_error}"
            ) from first_error
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            from inference.predictor import LimiXPredictor  # type: ignore

            return LimiXPredictor
        except Exception as second_error:
            raise LimiXUnavailableError(
                f"LimiX source found at {src}, but importing 'inference.predictor' failed: "
                f"{second_error}"
            ) from second_error


class LimiXScorer:
    """In-context probability-of-default scorer backed by LimiX-2.

    Args:
        model_path: Path to ``LimiX-2-1016.ckpt``. Resolved from ``LIMIX_MODEL_PATH`` and common
            cache locations when omitted.
        config_dir: Directory containing the V2 JSON configs. Resolved from ``LIMIX_CONFIG_DIR``
            and the installed LimiX package when omitted.
        task: ``"classification"`` or ``"regression"``.
        device: ``"cpu"``, ``"cuda"`` or ``"auto"``.
        n_pipelines: Number of ensemble pipelines from the V2 config. The released config uses 32
            members; 1-4 is sensible on CPU, 8-32 on GPU. ``None`` keeps all 32.
        seed: Random seed for preprocessing and in-context sampling.
        softmax_temperature: Temperature applied to model logits. ``None`` resolves to 0.9 for
            classification and 1.0 for regression; see the class docstring.
        cache_dir: LimiX preprocessing cache directory.
        mix_precision: Mixed precision on GPU. Forced off on CPU by LimiX.
        deterministic: Pin RNGs and deterministic kernels where available.
        verbose: Emit a warning when a CPU run uses many pipelines.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_dir: str | Path | None = None,
        *,
        task: Literal["classification", "regression"] = "classification",
        device: str = "auto",
        n_pipelines: int | None = 4,
        seed: int = 0,
        softmax_temperature: float | None = None,
        cache_dir: str | Path | None = None,
        mix_precision: bool = False,
        deterministic: bool = False,
        verbose: bool = True,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        if n_pipelines is not None and n_pipelines < 1:
            raise ValueError("n_pipelines must be positive or None.")
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.config_dir = Path(config_dir).expanduser() if config_dir else None
        self.task = task
        self.device = device
        self.n_pipelines = n_pipelines
        self.seed = seed
        if softmax_temperature is None:
            self.softmax_temperature = 0.9 if task == "classification" else 1.0
        else:
            self.softmax_temperature = float(softmax_temperature)
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir
            else Path.home() / ".cache" / "opencashflow" / "limix"
        )
        self.mix_precision = mix_precision
        self.deterministic = deterministic
        self.verbose = verbose
        self._predictor: Any = None
        self._imputation_predictor: Any = None
        self.feature_names_in_: list[str] = []
        self.n_features_in_: int = 0
        self.classes_: np.ndarray | None = None
        self.fill_values_: np.ndarray | None = None
        self.categories_: dict[str, pd.Index] = {}
        self.X_train_: np.ndarray | None = None
        self.X_train_raw_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.dataset_name_: str | None = None
        self._fitted = False

    # ------------------------------------------------------------------ availability
    @classmethod
    def availability(cls) -> dict[str, Any]:
        """Check whether the code, checkpoint and configs are discoverable, without loading weights."""
        model = _search_checkpoint()
        config = _search_config_dir()
        src = _resolve_limix_src()
        limix_installed = importlib.util.find_spec("limix") is not None
        available = bool(model and config)
        reasons: list[str] = []
        if model is None:
            reasons.append("checkpoint not found; set LIMIX_MODEL_PATH")
        if config is None:
            reasons.append("config directory not found; set LIMIX_CONFIG_DIR")
        if not limix_installed and src is None:
            reasons.append("LimiX source not installed; set LIMIX_SRC")
        return {
            "available": available,
            "model_path": str(model) if model else None,
            "config_dir": str(config) if config else None,
            "limix_src": str(src) if src else None,
            "limix_installed": limix_installed,
            "reasons": reasons,
        }

    @classmethod
    def is_available(cls) -> bool:
        """``True`` when a checkpoint and config directory can be resolved."""
        return bool(cls.availability()["available"])

    @staticmethod
    def license_notice() -> str:
        """Return the LimiX weights licence notice."""
        return LICENSE_NOTICE

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the scorer configuration."""
        info = self.availability()
        info.update(
            {
                "task": self.task,
                "device": self.device,
                "n_pipelines": self.n_pipelines,
                "seed": self.seed,
                "softmax_temperature": self.softmax_temperature,
                "fitted": self._fitted,
                "n_features_in": self.n_features_in_,
            }
        )
        return info

    # ------------------------------------------------------------------ sklearn-like API
    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> LimiXScorer:
        """Store the labelled in-context reference set.

        LimiX performs no gradient training. ``fit`` validates and caches the reference matrix and
        labels; everything else happens inside :meth:`predict_proba` or :meth:`predict`.
        """
        self._set_feature_names(X)
        X_prepared = self._prepare_X(X, fit=True)
        # Keep an unfilled copy for the imputation reference. Classification and regression keep
        # the median-filled matrix so their benchmark behaviour is unchanged.
        X_raw = self._prepare_X(X, fit=True, fill=False, dtype=np.float64)
        y_array = np.asarray(y).reshape(-1)
        if len(y_array) != len(X_prepared):
            raise ValueError(f"X has {len(X_prepared)} rows but y has {len(y_array)}.")
        if self.task == "classification":
            classes = np.unique(y_array)
            if not 2 <= classes.size <= 10:
                raise ValueError(f"LimiX classification supports 2-10 classes, got {classes.size}.")
            self.classes_ = classes
            self.y_train_ = y_array
        else:
            self.y_train_ = y_array.astype(float)
            self.classes_ = None
        self.X_train_ = X_prepared
        self.X_train_raw_ = X_raw
        self.dataset_name_ = _make_dataset_name(self.seed, X_prepared, y_array)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Class probabilities with shape ``(n_samples, n_classes)``."""
        if self.task != "classification":
            raise ValueError("predict_proba is only available when task='classification'.")
        self._require_fitted()
        x_query = self._prepare_X(X, fit=False)
        predictor = self._ensure_predictor()
        proba = predictor.predict(
            self.X_train_,
            self.y_train_,
            x_query,
            task_type="Classification",
            unique_dataset_name=self.dataset_name_
            or _make_dataset_name(self.seed, self.X_train_, self.y_train_),
        )
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        if proba.shape[1] != len(self.classes_):
            raise RuntimeError(
                f"LimiX returned {proba.shape[1]} columns but {len(self.classes_)} classes were fitted."
            )
        proba = np.clip(proba, 0.0, 1.0)
        totals = proba.sum(axis=1, keepdims=True)
        return proba / np.where(totals > 0, totals, 1.0)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicted labels for classification, or point predictions for regression."""
        if self.task == "regression":
            return self.predict_score(X)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_pd(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Probability of the positive (default) class for binary classification.

        The positive class is ``1`` when present, otherwise the largest label. Encode default as
        ``1`` to avoid ambiguity.
        """
        if self.task != "classification":
            raise ValueError("predict_pd requires task='classification'.")
        self._require_fitted()
        if len(self.classes_) != 2:  # type: ignore[arg-type]
            raise ValueError("predict_pd requires exactly two classes.")
        proba = self.predict_proba(X)
        positive = 1 if 1 in self.classes_ else int(self.classes_[-1])  # type: ignore[index]
        index = int(np.where(self.classes_ == positive)[0][0])  # type: ignore[index]
        return proba[:, index]

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Regression prediction on the original target scale."""
        if self.task != "regression":
            raise ValueError("predict_score requires task='regression'.")
        self._require_fitted()
        x_query = self._prepare_X(X, fit=False)
        predictor = self._ensure_predictor()
        prediction = predictor.predict(
            self.X_train_,
            self.y_train_,
            x_query,
            task_type="Regression",
            unique_dataset_name=self.dataset_name_
            or _make_dataset_name(self.seed, self.X_train_, self.y_train_),
        )
        return np.asarray(prediction, dtype=float).reshape(-1)

    def impute(
        self,
        X: pd.DataFrame | np.ndarray,
        reference: pd.DataFrame | np.ndarray | None = None,
    ) -> pd.DataFrame | np.ndarray:
        """Impute missing feature values with LimiX's feature-imputation task.

        Missing cells are passed to LimiX as NaN in both the query and the reference. Only those
        cells are replaced by LimiX's reconstruction; observed cells are returned unchanged. Median
        filling is deliberately skipped on this path so the model can use the missingness pattern.

        Args:
            X: Matrix with missing values (``NaN``).
            reference: Optional reference matrix, which may contain NaN. Defaults to the unfilled
                fitted training matrix, or to ``X`` itself when the scorer is not fitted.

        Returns:
            A matrix of the same shape as ``X`` with no NaN. A DataFrame is returned when ``X`` is a
            DataFrame.
        """
        is_frame = isinstance(X, pd.DataFrame)
        columns = list(X.columns) if is_frame else None
        index = X.index if is_frame else None
        x_query = self._prepare_X(X, fit=False, fill=False, dtype=np.float64)
        missing = np.isnan(x_query)
        if reference is None:
            ref = self.X_train_raw_ if self.X_train_raw_ is not None else x_query
            ref_y = self.y_train_ if self.y_train_ is not None else np.zeros(len(ref))
        else:
            ref = self._prepare_X(reference, fit=False, fill=False, dtype=np.float64)
            # An explicit reference carries no labels. Imputation does not use targets, so use
            # zeros of the exact reference length instead of slicing unrelated fitted labels.
            ref_y = np.zeros(len(ref))
        predictor = self._ensure_imputation_predictor()
        imputed = predictor.predict(
            ref,
            ref_y,
            x_query,
            task_type="Feature_imputation",
            unique_dataset_name=_make_dataset_name(self.seed, ref, ref_y),
        )
        if imputed is None:
            raise RuntimeError("LimiX returned no imputation output.")
        imputed = np.asarray(imputed, dtype=float)
        # LimiX returns train rows followed by query rows; keep the query block.
        if imputed.shape[0] == len(ref) + len(x_query):
            imputed = imputed[-len(x_query) :]
        if imputed.shape != x_query.shape:
            raise RuntimeError(
                f"Unexpected imputation shape {imputed.shape}, expected {x_query.shape}."
            )
        # LimiX reconstructs every cell. Keep its output only where the query was missing.
        result = x_query.astype(float, copy=True)
        result[missing] = imputed[missing]
        # A masked cell in a constant column can come back as NaN. Fall back to the learned
        # median, still touching missing positions only, so the result is always finite.
        residual = np.isnan(result)
        if residual.any():
            n_residual = int(residual.sum())
            warnings.warn(
                f"LimiX returned NaN for {n_residual} imputed cell(s); "
                "filled them from the training median.",
                UserWarning,
                stacklevel=2,
            )
            medians = (
                self.fill_values_ if self.fill_values_ is not None else np.zeros(result.shape[1])
            )
            fallback = np.broadcast_to(np.asarray(medians, dtype=float), result.shape)
            result[residual] = fallback[residual]
        if is_frame:
            return pd.DataFrame(result, columns=columns, index=index)
        return result

    # ------------------------------------------------------------------ internals
    def _set_feature_names(self, X: pd.DataFrame | np.ndarray) -> None:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [str(c) for c in X.columns]
        elif not self.feature_names_in_:
            array = np.asarray(X)
            self.feature_names_in_ = [f"f{i}" for i in range(array.shape[1])]
        self.n_features_in_ = len(self.feature_names_in_)

    def _prepare_X(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        fit: bool,
        fill: bool = True,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        """Encode a matrix for LimiX, optionally skipping the median fill.

        ``fill=False`` keeps NaN so the imputation task can see which cells are missing. Fill values
        are still learned from the data on the fit path, so the classification matrix is unchanged.
        ``dtype`` is float32 for inference; imputation requests float64 so observed cells survive
        the round trip unchanged.
        """
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = [str(c) for c in frame.columns]
            if fit or not self.feature_names_in_:
                self.feature_names_in_ = list(frame.columns)
                self.n_features_in_ = len(self.feature_names_in_)
                self.categories_ = {}
                for column in frame.columns:
                    if _is_categorical(frame[column]):
                        categories = pd.Index(sorted(frame[column].dropna().astype(str).unique()))
                        self.categories_[column] = categories
            elif self.feature_names_in_:
                frame = frame.reindex(columns=self.feature_names_in_)
            encoded_data: dict[str, Any] = {}
            for column in self.feature_names_in_:
                if column not in frame.columns:
                    encoded_data[column] = np.nan
                elif column in self.categories_:
                    categories = self.categories_[column]
                    codes = pd.Categorical(
                        frame[column].astype(str), categories=categories
                    ).codes.astype(float)
                    if not fill:
                        # astype(str) turns NaN into the string "nan", which would otherwise be
                        # encoded as an unseen category (-1) instead of a missing cell.
                        codes[frame[column].isna().to_numpy()] = np.nan
                    encoded_data[column] = codes
                else:
                    encoded_data[column] = pd.to_numeric(frame[column], errors="coerce")
            numeric = pd.DataFrame(encoded_data, index=frame.index)
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError(f"Expected a 2-D matrix, got shape {array.shape}.")
            if fit or not self.feature_names_in_:
                self.feature_names_in_ = [f"f{i}" for i in range(array.shape[1])]
                self.n_features_in_ = array.shape[1]
            if array.shape[1] != len(self.feature_names_in_):
                raise ValueError(
                    f"Expected {len(self.feature_names_in_)} features, got {array.shape[1]}."
                )
            numeric = pd.DataFrame(array, columns=self.feature_names_in_)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        if fit or self.fill_values_ is None:
            medians = numeric.median(axis=0, skipna=True).to_numpy(dtype=float)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            self.fill_values_ = medians
        if not fill:
            return np.ascontiguousarray(numeric.to_numpy(dtype=dtype))
        filled = numeric.fillna(pd.Series(self.fill_values_, index=self.feature_names_in_))
        return np.ascontiguousarray(filled.to_numpy(dtype=dtype))

    def _build_predictor(self, config_name: str, cache_subdir: str = "") -> Any:
        model_path = self.model_path or _search_checkpoint()
        if model_path is None or not Path(model_path).is_file():
            raise LimiXUnavailableError(
                "LimiX checkpoint not found. Set LIMIX_MODEL_PATH to LimiX-2-1016.ckpt. "
                f"Searched: {self.availability()['reasons']}"
            )
        config_dir = self.config_dir or _search_config_dir()
        if config_dir is None:
            raise LimiXUnavailableError(
                "LimiX config directory not found. Set LIMIX_CONFIG_DIR to the LimiX 'config' folder."
            )
        config_path = Path(config_dir) / config_name
        if not config_path.is_file():
            raise LimiXUnavailableError(f"LimiX config not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pipelines = config.get("pipelines", [])
        if self.n_pipelines is not None:
            config["pipelines"] = pipelines[: min(self.n_pipelines, len(pipelines))]
        if not config["pipelines"]:
            raise LimiXUnavailableError(f"LimiX config {config_path} has no pipelines.")

        import torch

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise LimiXUnavailableError(
                "CUDA was requested but torch.cuda.is_available() is False."
            )
        if device == "cpu" and self.verbose and len(config["pipelines"]) > 8:
            warnings.warn(
                "Running LimiX on CPU with more than 8 pipelines is slow. "
                "Use n_pipelines=1-4 for experiments.",
                stacklevel=2,
            )

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        cache_dir = self.cache_dir / cache_subdir if cache_subdir else self.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        predictor_class = _load_predictor_class()
        return predictor_class(
            device=torch.device(device),
            model_path=str(model_path),
            inference_config=config,
            mix_precision=self.mix_precision and device != "cpu",
            seed=self.seed,
            softmax_temperature=self.softmax_temperature,
            deterministic=self.deterministic,
            cache_dir=str(cache_dir),
        )

    def _ensure_predictor(self) -> Any:
        if self._predictor is None:
            config_name = (
                CLASSIFICATION_CONFIG if self.task == "classification" else REGRESSION_CONFIG
            )
            self._predictor = self._build_predictor(config_name)
        return self._predictor

    def _ensure_imputation_predictor(self) -> Any:
        if self._imputation_predictor is None:
            self._imputation_predictor = self._build_predictor(
                IMPUTATION_CONFIG, cache_subdir="imputation"
            )
        return self._imputation_predictor

    def _require_fitted(self) -> None:
        if not self._fitted or self.X_train_ is None:
            raise RuntimeError("LimiXScorer is not fitted. Call fit(X, y) first.")

    def close(self) -> None:
        """Release the in-memory predictor and free memory."""
        self._predictor = None
        self._imputation_predictor = None
        gc.collect()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_predictor"] = None
        state["_imputation_predictor"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


__all__ = [
    "DEFAULT_CHECKPOINT",
    "LICENSE_NOTICE",
    "LimiXScorer",
    "LimiXUnavailableError",
]

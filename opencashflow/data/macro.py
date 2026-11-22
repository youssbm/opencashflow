"""Optional macro-context features from Eurostat, EBA or a national source.

Macro indicators are useful for stress testing and portfolio monitoring, but they are not a
substitute for borrower-level cash flow. This module provides a small, explicit joiner that:

* accepts a tidy long table (country, period, indicator, value);
* normalises periods to months;
* forward-fills within country;
* applies a publication lag and configurable lags so the value is known at the scoring date;
* optionally computes year-on-year changes;
* records whether a macro lookup was missing.

Official series are released after their reference period. ``publication_lag`` (one month by
default) shifts the lookup back so lag 0 is the latest value published by the scoring date.

The data itself is not bundled. Download the series you need from Eurostat's REST API, the EBA data
portal or a national statistics office, store the extraction date and vintage, and keep the raw
file outside version control. In the example the 15 February score reads the January value.

Example:
    >>> import pandas as pd
    >>> macro = pd.DataFrame({
    ...     "country": ["DE", "DE", "PL", "PL"],
    ...     "period": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
    ...     "indicator": ["confidence"] * 4,
    ...     "value": [96.0, 95.0, 98.0, 97.0],
    ... })
    >>> joiner = MacroFeatureJoiner(lags=(0, 1), yoy_periods=None).fit(macro)
    >>> features = pd.DataFrame(index=["A", "B"])
    >>> countries = pd.Series({"A": "DE", "B": "PL"})
    >>> as_of = pd.Series({"A": "2024-02-15", "B": "2024-02-15"})
    >>> out = joiner.transform(features, countries, as_of)
    >>> float(out.loc["A", "macro_confidence_lag0"])
    96.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def load_macro_csv(
    path: str,
    *,
    country_col: str = "country",
    period_col: str = "period",
    indicator_col: str = "indicator",
    value_col: str = "value",
) -> pd.DataFrame:
    """Load a tidy macro CSV and validate the required columns."""
    frame = pd.read_csv(path)
    required = {country_col, period_col, indicator_col, value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Macro CSV is missing columns: {sorted(missing)}")
    return frame.rename(
        columns={
            country_col: "country",
            period_col: "period",
            indicator_col: "indicator",
            value_col: "value",
        }
    )


@dataclass
class MacroFeatureJoiner:
    """Join lagged macro indicators to an entity-level feature matrix.

    Args:
        lags: Month lags to include, measured back from the publication-shifted scoring month.
            ``0`` is the latest published observation, ``3`` is three months earlier, and so on.
        publication_lag: Months between a series' reference period and its release. The lookup uses
            ``as_of - publication_lag``, so lag 0 is the latest value published by the scoring
            date. Must be non-negative.
        yoy_periods: Number of periods for a year-on-year change column. ``12`` for monthly data,
            ``4`` for quarterly data, or ``None`` to skip.
        prefix: Prefix for generated feature names.
        fill_missing: Value used when no observation exists at or before the effective month. The
            accompanying ``macro_missing`` flag records the event.
        add_missing_flag: Add a 0/1 column indicating any missing macro lookup.
    """

    lags: tuple[int, ...] = (0, 3, 12)
    publication_lag: int = 1
    yoy_periods: int | None = 12
    prefix: str = "macro"
    fill_missing: float = 0.0
    add_missing_flag: bool = True

    def __post_init__(self) -> None:
        if not self.lags:
            raise ValueError("At least one lag is required.")
        if any(lag < 0 for lag in self.lags):
            raise ValueError("Lags must be non-negative.")
        if self.publication_lag < 0:
            raise ValueError("publication_lag must be non-negative.")
        if self.yoy_periods is not None and self.yoy_periods < 1:
            raise ValueError("yoy_periods must be positive or None.")

    def fit(
        self,
        macro: pd.DataFrame,
        *,
        country_col: str = "country",
        period_col: str = "period",
        indicator_col: str = "indicator",
        value_col: str = "value",
    ) -> MacroFeatureJoiner:
        """Learn the indicator set and build a complete monthly panel by country."""
        required = {country_col, period_col, indicator_col, value_col}
        missing = required - set(macro.columns)
        if missing:
            raise ValueError(f"Macro table is missing columns: {sorted(missing)}")
        frame = macro.copy()
        frame["country"] = frame[country_col].astype(str)
        frame["period"] = pd.PeriodIndex(pd.to_datetime(frame[period_col]), freq="M")
        wide = frame.pivot_table(
            index=["country", "period"],
            columns=frame[indicator_col].astype(str),
            values=value_col,
            aggfunc="last",
        ).sort_index()
        if wide.empty:
            raise ValueError("Macro table produced an empty panel.")
        self.indicators_ = [str(c) for c in wide.columns]
        full_periods = pd.period_range(
            wide.index.get_level_values("period").min(),
            wide.index.get_level_values("period").max(),
            freq="M",
        )
        countries = wide.index.get_level_values("country").unique()
        panel = (
            wide.reindex(
                pd.MultiIndex.from_product([countries, full_periods], names=["country", "period"])
            )
            .groupby(level="country")
            .ffill()
        )
        self.panel_ = panel
        return self

    def transform(
        self,
        features: pd.DataFrame,
        countries: Mapping[str, str] | pd.Series,
        as_of: pd.Timestamp | str | pd.Series,
    ) -> pd.DataFrame:
        """Return ``features`` with lagged macro columns appended.

        Args:
            features: Entity-level feature frame indexed by ``entity_id``.
            countries: Mapping or Series from entity id to ISO country code.
            as_of: Scoring date, scalar or Series indexed by entity id. The lookup uses the latest
                value published by this date, shifting back by ``publication_lag`` months.
        """
        if not hasattr(self, "panel_"):
            raise RuntimeError("MacroFeatureJoiner is not fitted. Call fit() first.")
        entity_ids = features.index
        country_series = (
            pd.Series(countries, index=entity_ids)
            if not isinstance(countries, pd.Series)
            else countries.reindex(entity_ids)
        )
        if country_series.isna().any():
            raise ValueError("Every entity needs a country mapping.")
        as_of_series = (
            pd.Series(as_of, index=entity_ids)
            if np.isscalar(as_of) or isinstance(as_of, (str, pd.Timestamp))
            else pd.Series(as_of).reindex(entity_ids)
        )
        periods = pd.PeriodIndex(pd.to_datetime(as_of_series), freq="M") - self.publication_lag
        lagged_panels: dict[int, pd.DataFrame] = {}
        for lag in self.lags:
            lagged_panels[lag] = self.panel_.groupby(level="country").shift(lag)
        yoy_panel = None
        if self.yoy_periods is not None:
            yoy_panel = (
                self.panel_ / self.panel_.groupby(level="country").shift(self.yoy_periods) - 1.0
            )

        rows: list[dict[str, Any]] = []
        missing_flags: list[float] = []
        for country, period in zip(country_series, periods):
            row: dict[str, Any] = {}
            missing = 0
            for lag in self.lags:
                panel = lagged_panels[lag]
                for indicator in self.indicators_:
                    key = (str(country), period)
                    value = panel[indicator].get(key, np.nan) if key in panel.index else np.nan
                    if not np.isfinite(value):
                        # Fall back to the latest available observation before the period.
                        if str(country) not in panel.index.get_level_values(0):
                            value = np.nan
                        else:
                            history = panel[indicator].loc[str(country)]
                            history = history.loc[history.index <= period].dropna()
                            value = float(history.iloc[-1]) if len(history) else np.nan
                    if not np.isfinite(value):
                        missing += 1
                        value = self.fill_missing
                    row[f"{self.prefix}_{indicator}_lag{lag}"] = float(value)
            if yoy_panel is not None:
                for indicator in self.indicators_:
                    key = (str(country), period)
                    value = (
                        yoy_panel[indicator].get(key, np.nan) if key in yoy_panel.index else np.nan
                    )
                    if not np.isfinite(value):
                        missing += 1
                        value = self.fill_missing
                    row[f"{self.prefix}_{indicator}_yoy{self.yoy_periods}"] = float(value)
            rows.append(row)
            missing_flags.append(float(missing > 0))
        macro_frame = pd.DataFrame(rows, index=entity_ids)
        if self.add_missing_flag:
            macro_frame[f"{self.prefix}_missing"] = missing_flags
        return features.join(macro_frame)

    def fit_transform(
        self, macro: pd.DataFrame, features: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Fit on a macro table and transform a feature matrix in one call."""
        return self.fit(macro).transform(features, **kwargs)


__all__ = ["MacroFeatureJoiner", "load_macro_csv"]

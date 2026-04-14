"""Functions for learning table specs from CSV files.

Provides learn_from_csv for automatic distribution fitting
and template_from_data for generating editable spec skeletons.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from pygmalion.io.fitting import fit_best_distribution


_CATEGORICAL_THRESHOLD = 10


def _is_categorical(series: pd.Series) -> bool:
    """Determine if a pandas Series should be treated as categorical.

    A series is categorical if its dtype is object (strings),
    if it has 10 or fewer unique values, or if its standard
    deviation is zero.

    Args:
        series: A pandas Series to analyze.

    Returns:
        True if the series should be treated as categorical.
    """
    if series.dtype == object:
        return True
    if series.nunique() <= _CATEGORICAL_THRESHOLD:
        return True
    if series.std() == 0:
        return True
    return False


def _analyze_numeric(series: pd.Series) -> dict:
    """Fit a normal distribution to a numeric series.

    Args:
        series: A numeric pandas Series.

    Returns:
        A column spec dictionary with type "normal", fitted
        mean, std, min, and max.
    """
    return {
        "name": series.name,
        "type": "normal",
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
    }


def _analyze_categorical(series: pd.Series) -> dict:
    """Extract categories and frequency weights from a series.

    Args:
        series: A pandas Series with categorical values.

    Returns:
        A column spec dictionary with type "categorical",
        values, and weights that sum to 1.0.
    """
    freq = series.value_counts(normalize=True)
    values = freq.index.astype(str).tolist()
    weights = [round(float(w), 4) for w in freq.values]

    remainder = round(1.0 - sum(weights), 4)
    if abs(remainder) > 0:
        weights[-1] = round(weights[-1] + remainder, 4)

    return {
        "name": series.name,
        "type": "categorical",
        "values": values,
        "weights": weights,
    }


def _analyze_bootstrap(series: pd.Series) -> dict:
    """Create a bootstrap spec from a series' observed values.

    Args:
        series: A pandas Series with the observed values.

    Returns:
        A column spec dictionary with type "bootstrap" and
        the list of observed values.
    """
    values = series.tolist()
    return {
        "name": series.name,
        "type": "bootstrap",
        "values": values,
    }


def learn_from_csv(
    path: str | Path,
    num_rows: int | None = None,
    strategy: str = "parametric",
) -> dict:
    """Learn a table spec from a CSV file.

    Reads a CSV file, analyzes each column, and produces a JSON
    spec dictionary ready to use with synthesize().

    Three strategies are available:

    - "parametric": fits normal distributions for numeric columns
      and frequency-weighted categoricals. Assumes normality.
    - "bootstrap": stores observed values for resampling. Makes no
      distributional assumptions.
    - "auto_fit": tries multiple distributions per numeric column,
      evaluates with AIC and KS test, selects the best fit.
      Falls back to bootstrap if no distribution fits well.

    Args:
        path: Path to the CSV file.
        num_rows: Number of rows for the generated spec. If None,
            uses the number of rows in the original CSV.
        strategy: Learning strategy. "parametric" (default),
            "bootstrap", or "auto_fit".

    Returns:
        A dictionary with 'num_rows' and 'columns' keys, valid
        as input to synthesize().

    Raises:
        ValueError: If strategy is not valid.

    Example:
        >>> spec = learn_from_csv("data.csv", strategy="auto_fit")
        >>> df = synthesize(spec)
    """
    if strategy not in ("parametric", "bootstrap", "auto_fit"):
        raise ValueError(
            f"strategy debe ser 'parametric', 'bootstrap' o 'auto_fit', "
            f"recibió: '{strategy}'"
        )

    df = pd.read_csv(str(path))

    if num_rows is None:
        num_rows = len(df)

    columns = []
    for col_name in df.columns:
        series = df[col_name].dropna()

        if len(series) == 0:
            continue

        if strategy == "bootstrap":
            columns.append(_analyze_bootstrap(series))
        elif strategy == "auto_fit":
            if _is_categorical(series):
                columns.append(_analyze_categorical(series))
            else:
                columns.append(fit_best_distribution(series))
        else:
            if _is_categorical(series):
                columns.append(_analyze_categorical(series))
            else:
                columns.append(_analyze_numeric(series))

    return {
        "num_rows": num_rows,
        "columns": columns,
    }


def _template_numeric(series: pd.Series) -> dict:
    """Create a simplified numeric column template.

    Similar to _analyze_numeric but rounds values to integers
    for readability when editing manually.

    Args:
        series: A numeric pandas Series.

    Returns:
        A column spec dictionary with rounded parameters.
    """
    return {
        "name": series.name,
        "type": "normal",
        "mean": round(float(series.mean())),
        "std": max(round(float(series.std())), 1),
        "min": round(float(series.min())),
        "max": round(float(series.max())),
    }


def _template_categorical(series: pd.Series) -> dict:
    """Create a simplified categorical column template.

    Unlike _analyze_categorical, omits weights (assumes uniform
    distribution) and sorts values alphabetically.

    Args:
        series: A pandas Series with categorical values.

    Returns:
        A column spec dictionary with sorted values, no weights.
    """
    values = series.dropna().astype(str).unique().tolist()
    values.sort()
    return {
        "name": series.name,
        "type": "categorical",
        "values": values,
    }


def template_from_data(path: str | Path, num_rows: int = 1000) -> dict:
    """Generate an editable template spec from a CSV file.

    Similar to learn_from_csv but produces a simpler, more readable
    spec intended for manual editing. Numeric parameters are rounded
    to integers, and categorical columns omit weights (uniform
    distribution by default).

    Args:
        path: Path to the CSV file.
        num_rows: Number of rows for the template. Defaults to 1000.

    Returns:
        A dictionary with 'num_rows' and 'columns' keys, valid
        as input to synthesize() after optional editing.

    Example:
        >>> from pygmalion import template_from_data
        >>> template = template_from_data("data.csv")
        >>> # Edit template as needed, then:
        >>> df = synthesize(template)
    """
    df = pd.read_csv(str(path))

    columns = []
    for col_name in df.columns:
        series = df[col_name].dropna()

        if len(series) == 0:
            continue

        if _is_categorical(series):
            columns.append(_template_categorical(series))
        else:
            columns.append(_template_numeric(series))

    return {
        "num_rows": num_rows,
        "columns": columns,
    }
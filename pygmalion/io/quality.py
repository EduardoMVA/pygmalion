"""Quality evaluation comparing real and synthetic DataFrames.

Provides column-level and overall similarity metrics using
statistical tests and frequency comparisons.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def _detect_column_type(real: pd.Series, synthetic: pd.Series) -> str:
    """Determine if a column pair should be compared as numeric or categorical.

    Args:
        real: Column from the real DataFrame.
        synthetic: Column from the synthetic DataFrame.

    Returns:
        "numeric" or "categorical".
    """
    if real.dtype == object or synthetic.dtype == object:
        return "categorical"
    if real.nunique() <= 10 and synthetic.nunique() <= 10:
        return "categorical"
    return "numeric"


def _compare_numeric(real: pd.Series, synthetic: pd.Series) -> dict:
    """Compare two numeric columns using descriptive stats and KS test.

    Args:
        real: Numeric column from the real DataFrame.
        synthetic: Numeric column from the synthetic DataFrame.

    Returns:
        Dictionary with real/synthetic stats, differences,
        KS statistic, p-value, and a score from 0 to 1.
    """
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(real.dropna(), synthetic.dropna())

    return {
        "type": "numeric",
        "real": {
            "mean": round(float(real.mean()), 4),
            "std": round(float(real.std()), 4),
            "min": round(float(real.min()), 4),
            "max": round(float(real.max()), 4),
        },
        "synthetic": {
            "mean": round(float(synthetic.mean()), 4),
            "std": round(float(synthetic.std()), 4),
            "min": round(float(synthetic.min()), 4),
            "max": round(float(synthetic.max()), 4),
        },
        "mean_diff": round(abs(float(real.mean()) - float(synthetic.mean())), 4),
        "std_diff": round(abs(float(real.std()) - float(synthetic.std())), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "score": round(1.0 - float(ks_stat), 4),
    }


def _compare_categorical(real: pd.Series, synthetic: pd.Series) -> dict:
    """Compare two categorical columns by frequency distributions.

    Args:
        real: Categorical column from the real DataFrame.
        synthetic: Categorical column from the synthetic DataFrame.

    Returns:
        Dictionary with category details, average frequency
        difference, and a score from 0 to 1.
    """
    real_freq = real.value_counts(normalize=True)
    synth_freq = synthetic.value_counts(normalize=True)

    all_values = set(real_freq.index) | set(synth_freq.index)
    freq_diffs = []
    details = {}

    for val in all_values:
        r = float(real_freq.get(val, 0))
        s = float(synth_freq.get(val, 0))
        diff = abs(r - s)
        freq_diffs.append(diff)
        details[str(val)] = {
            "real_freq": round(r, 4),
            "synthetic_freq": round(s, 4),
            "diff": round(diff, 4),
        }

    avg_diff = float(np.mean(freq_diffs))

    return {
        "type": "categorical",
        "num_real_categories": len(real_freq),
        "num_synthetic_categories": len(synth_freq),
        "category_details": details,
        "avg_freq_diff": round(avg_diff, 4),
        "score": round(1.0 - avg_diff, 4),
    }


def quality_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> dict:
    """Compare real and synthetic DataFrames and produce a quality report.

    For each common column, computes similarity metrics:
    - Numeric columns: Kolmogorov-Smirnov test, mean/std differences.
    - Categorical columns: frequency distribution differences.

    Each column receives a score from 0 (completely different)
    to 1 (statistically identical). The overall score is the
    average across all columns.

    Args:
        real: The original (real) DataFrame. Must be a pandas DataFrame.
        synthetic: The synthetic DataFrame to evaluate. Must be a
            pandas DataFrame.

    Returns:
        A dictionary with keys:
            - 'num_columns_compared': number of columns analyzed.
            - 'overall_score': average quality score (0 to 1).
            - 'columns': dict mapping column names to detailed reports.

    Raises:
        ValueError: If there are no common columns between the
            two DataFrames.

    Example:
        >>> from pygmalion import quality_report
        >>> report = quality_report(df_real, df_synthetic)
        >>> print(report["overall_score"])
        0.92
    """
    common_columns = [col for col in real.columns if col in synthetic.columns]

    if not common_columns:
        raise ValueError("No hay columnas en común entre los DataFrames")

    column_reports = {}
    scores = []

    for col_name in common_columns:
        col_type = _detect_column_type(real[col_name], synthetic[col_name])

        if col_type == "numeric":
            report = _compare_numeric(real[col_name], synthetic[col_name])
        else:
            report = _compare_categorical(real[col_name], synthetic[col_name])

        column_reports[col_name] = report
        scores.append(report["score"])

    return {
        "num_columns_compared": len(common_columns),
        "overall_score": round(float(np.mean(scores)), 4),
        "columns": column_reports,
    }
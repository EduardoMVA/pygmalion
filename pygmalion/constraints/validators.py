"""Post-generation constraint validation and filtering.

Evaluates boolean expressions against generated DataFrames
and filters out rows that violate the constraints.
"""

import numpy as np
import polars as pl


def _evaluate_constraint(df: pl.DataFrame, constraint: str) -> np.ndarray:
    """Evaluate a single constraint expression against a DataFrame.

    Args:
        df: The Polars DataFrame to evaluate against.
        constraint: A boolean expression string using column
            names as variables (e.g., "age >= 18").

    Returns:
        A boolean numpy array where True means the row satisfies
        the constraint.
    """
    variables = {col: df[col].to_numpy() for col in df.columns}
    result = eval(constraint, {"__builtins__": {}}, variables)
    return np.asarray(result, dtype=bool)


def apply_constraints(
    df: pl.DataFrame,
    constraints: list[str],
) -> pl.DataFrame:
    """Filter a DataFrame by applying all constraints.

    Evaluates each constraint and keeps only rows that satisfy
    all of them (logical AND).

    Args:
        df: The Polars DataFrame to filter.
        constraints: List of boolean expression strings.

    Returns:
        A filtered Polars DataFrame with only valid rows.
    """
    if not constraints:
        return df

    mask = np.ones(len(df), dtype=bool)
    for constraint in constraints:
        result = _evaluate_constraint(df, constraint)
        mask = mask & result

    return df.filter(pl.Series(mask))
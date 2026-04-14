"""Functions for exporting DataFrames to files."""

from pathlib import Path
import json

import polars as pl
import pandas as pd


def _ensure_polars(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Convert a Pandas DataFrame to Polars if needed.

    Args:
        df: A Polars or Pandas DataFrame.

    Returns:
        A Polars DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def to_csv(df: pl.DataFrame | pd.DataFrame, path: str | Path) -> None:
    """Export a DataFrame to a CSV file.

    Accepts both Polars and Pandas DataFrames. If a Pandas
    DataFrame is provided, it is converted to Polars internally
    before writing.

    Args:
        df: The DataFrame to export.
        path: Destination file path.

    Example:
        >>> from pygmalion import synthesize, to_csv
        >>> df = synthesize(spec)
        >>> to_csv(df, "output.csv")
    """
    df = _ensure_polars(df)
    df.write_csv(str(path))


def to_json(df: pl.DataFrame | pd.DataFrame, path: str | Path) -> None:
    """Export a DataFrame to a JSON file in records format.

    Each row becomes a JSON object. The output is a JSON array
    of objects, formatted with indentation for readability.
    Supports Unicode characters (accents, ñ, etc.).

    Accepts both Polars and Pandas DataFrames.

    Args:
        df: The DataFrame to export.
        path: Destination file path.

    Example:
        >>> from pygmalion import synthesize, to_json
        >>> df = synthesize(spec)
        >>> to_json(df, "output.json")
    """
    df = _ensure_polars(df)
    records = df.to_dicts()
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
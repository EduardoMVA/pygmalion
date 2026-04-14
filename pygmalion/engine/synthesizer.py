"""Engine that orchestrates data generation from validated specs.

This module connects the schema layer with the generators layer,
resolving column dependencies and assembling the final DataFrame.
"""

import numpy as np
import polars as pl
import pandas as pd

import pygmalion.generators  # noqa: F401 - registra los generadores

from pygmalion.generators.registry import get_generator
from pygmalion.schema.spec import TableSpec, DerivedColumnSpec, ConditionalColumnSpec
from pygmalion.constraints.validators import apply_constraints


def _generate_independent_columns(
    spec: TableSpec, num_rows: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Generate all columns that don't depend on other columns."""
    data = {}
    for col in spec.columns:
        if isinstance(col, (DerivedColumnSpec, ConditionalColumnSpec)):
            continue
        gen_class = get_generator(col.type)
        gen = gen_class(col)
        data[col.name] = gen.generate(num_rows, rng)
    return data


def _resolve_dependent_columns(
    spec: TableSpec, data: dict[str, np.ndarray], num_rows: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Resolve derived and conditional columns in dependency order."""
    dependent = [
        col for col in spec.columns
        if isinstance(col, (DerivedColumnSpec, ConditionalColumnSpec))
    ]
    max_iterations = len(dependent) + 1
    iteration = 0

    while dependent and iteration < max_iterations:
        remaining = []
        for col in dependent:
            if isinstance(col, DerivedColumnSpec):
                deps = col.dependencies
            else:
                deps = [col.condition_column]

            missing = [dep for dep in deps if dep not in data]
            if missing:
                remaining.append(col)
            else:
                if isinstance(col, DerivedColumnSpec):
                    variables = {dep: data[dep] for dep in col.dependencies}
                    data[col.name] = eval(col.expr, {"__builtins__": {}}, variables)
                else:
                    gen_class = get_generator(col.type)
                    gen = gen_class(col)
                    data[col.name] = gen.generate(num_rows, rng, context=data)

        dependent = remaining
        iteration += 1

    if dependent:
        names = [col.name for col in dependent]
        raise ValueError(
            f"No se pudieron resolver las dependencias para: {names}"
        )

    return data


def _build_dataframe(
    spec: TableSpec, data: dict[str, np.ndarray]
) -> pl.DataFrame:
    """Assemble a Polars DataFrame preserving the column order from the spec."""
    ordered = {col.name: data[col.name] for col in spec.columns}
    return pl.DataFrame(ordered)


def _generate_batch(
    spec: TableSpec, num_rows: int, rng: np.random.Generator
) -> pl.DataFrame:
    """Generate a single batch of rows."""
    data = _generate_independent_columns(spec, num_rows, rng)
    data = _resolve_dependent_columns(spec, data, num_rows, rng)
    return _build_dataframe(spec, data)


def synthesize(
    spec: dict | TableSpec,
    output_format: str = "pandas",
    max_attempts: int = 10,
    oversample_factor: float = 1.5,
) -> pd.DataFrame | pl.DataFrame:
    """Generate a synthetic DataFrame from a JSON spec.

    Args:
        spec: Table specification. Can be a dictionary or TableSpec.
        output_format: 'pandas' (default) or 'polars'.
        max_attempts: Maximum generation cycles for constraints.
        oversample_factor: Multiplier for batch size during
            constraint rejection sampling.

    Returns:
        A DataFrame with the generated synthetic data.

    Raises:
        ValidationError: If the spec is invalid.
        RuntimeError: If constraints cannot be satisfied.
        ValueError: If output_format is invalid.
    """
    if isinstance(spec, dict):
        spec = TableSpec(**spec)

    rng = np.random.default_rng(spec.seed)

    if not spec.constraints:
        df = _generate_batch(spec, spec.num_rows, rng)
    else:
        collected = []
        total_collected = 0
        attempts = 0

        while total_collected < spec.num_rows and attempts < max_attempts:
            rows_needed = spec.num_rows - total_collected
            batch_size = int(rows_needed * oversample_factor)
            batch_size = max(batch_size, rows_needed)

            batch = _generate_batch(spec, batch_size, rng)
            batch = apply_constraints(batch, spec.constraints)

            collected.append(batch)
            total_collected += len(batch)
            attempts += 1

        if total_collected < spec.num_rows:
            raise RuntimeError(
                f"No se alcanzaron {spec.num_rows} filas después de "
                f"{max_attempts} intentos. Se obtuvieron {total_collected}. "
                f"Las constraints pueden ser demasiado restrictivas."
            )

        df = pl.concat(collected)
        df = df.head(spec.num_rows)

    if output_format == "pandas":
        return df.to_pandas()
    elif output_format == "polars":
        return df
    else:
        raise ValueError(
            f"output_format debe ser 'pandas' o 'polars', recibió: '{output_format}'"
        )
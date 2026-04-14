"""Statistical summary of specs without data generation.

Provides stats_only() for inspecting expected column
statistics directly from the spec definition.
"""

import numpy as np

from pygmalion.schema.spec import (
    TableSpec,
    NormalColumnSpec,
    UniformColumnSpec,
    CategoricalColumnSpec,
    MixtureColumnSpec,
    DerivedColumnSpec,
    ConditionalColumnSpec,
    LognormalColumnSpec,
    BetaColumnSpec,
    GammaColumnSpec,
    ExponentialColumnSpec,
    ParetoColumnSpec,
    StudentTColumnSpec,
    PoissonColumnSpec,
    BinomialColumnSpec,
)


def _stats_normal(col: NormalColumnSpec) -> dict:
    """Extract stats from a normal column spec.

    Args:
        col: A validated NormalColumnSpec.

    Returns:
        Dictionary with type, mean, std, and optional min/max.
    """
    stats = {
        "type": "normal",
        "mean": col.mean,
        "std": col.std,
    }
    if col.min is not None:
        stats["min"] = col.min
    if col.max is not None:
        stats["max"] = col.max
    return stats


def _stats_beta(col: BetaColumnSpec) -> dict:
    """Extract stats from a beta column spec.

    Args:
        col: A validated BetaColumnSpec.

    Returns:
        Dictionary with type, alpha, beta_param, range,
        and expected mean.
    """
    raw_mean = col.alpha / (col.alpha + col.beta_param)
    scaled_mean = col.low + raw_mean * (col.high - col.low)
    return {
        "type": "beta",
        "alpha": col.alpha,
        "beta_param": col.beta_param,
        "low": col.low,
        "high": col.high,
        "expected_mean": round(scaled_mean, 4),
    }


def _stats_gamma(col: GammaColumnSpec) -> dict:
    """Extract stats from a gamma column spec.

    Args:
        col: A validated GammaColumnSpec.

    Returns:
        Dictionary with type, shape, scale, expected mean,
        and expected variance.
    """
    return {
        "type": "gamma",
        "shape": col.shape,
        "scale": col.scale,
        "expected_mean": round(col.shape * col.scale, 4),
        "expected_variance": round(col.shape * col.scale ** 2, 4),
    }


def _stats_exponential(col: ExponentialColumnSpec) -> dict:
    """Extract stats from an exponential column spec.

    Args:
        col: A validated ExponentialColumnSpec.

    Returns:
        Dictionary with type, scale, rate, and expected mean.
    """
    return {
        "type": "exponential",
        "scale": col.scale,
        "rate": round(1.0 / col.scale, 4),
        "expected_mean": col.scale,
    }


def _stats_pareto(col: ParetoColumnSpec) -> dict:
    """Extract stats from a Pareto column spec.

    Args:
        col: A validated ParetoColumnSpec.

    Returns:
        Dictionary with type, alpha, scale, expected mean
        (if defined), and minimum value.
    """
    stats = {
        "type": "pareto",
        "alpha": col.alpha,
        "scale": col.scale,
        "min_value": col.scale,
    }
    if col.alpha > 1:
        stats["expected_mean"] = round(
            col.scale * col.alpha / (col.alpha - 1), 4
        )
    else:
        stats["expected_mean"] = "infinite (alpha <= 1)"
    return stats

def _stats_student_t(col: StudentTColumnSpec) -> dict:
    """Extract stats from a Student's t column spec.

    Args:
        col: A validated StudentTColumnSpec.

    Returns:
        Dictionary with type, df, loc, scale, and expected
        mean (if defined).
    """
    result = {
        "type": "student_t",
        "df": col.df,
        "loc": col.loc,
        "scale": col.scale,
    }
    if col.df > 1:
        result["expected_mean"] = col.loc
    else:
        result["expected_mean"] = "undefined (df <= 1)"
    if col.df > 2:
        variance = col.scale ** 2 * col.df / (col.df - 2)
        result["expected_variance"] = round(variance, 4)
    else:
        result["expected_variance"] = "infinite (df <= 2)"
    return result


def _stats_poisson(col: PoissonColumnSpec) -> dict:
    """Extract stats from a Poisson column spec.

    Args:
        col: A validated PoissonColumnSpec.

    Returns:
        Dictionary with type, mu, expected mean and variance.
    """
    return {
        "type": "poisson",
        "mu": col.mu,
        "expected_mean": col.mu,
        "expected_variance": col.mu,
    }


def _stats_binomial(col: BinomialColumnSpec) -> dict:
    """Extract stats from a binomial column spec.

    Args:
        col: A validated BinomialColumnSpec.

    Returns:
        Dictionary with type, n, p, expected mean and variance.
    """
    return {
        "type": "binomial",
        "n": col.n,
        "p": col.p,
        "expected_mean": round(col.n * col.p, 4),
        "expected_variance": round(col.n * col.p * (1 - col.p), 4),
    }


def _stats_uniform(col: UniformColumnSpec) -> dict:
    """Extract stats from a uniform column spec.

    Args:
        col: A validated UniformColumnSpec.

    Returns:
        Dictionary with type, low, high, and expected mean.
    """
    return {
        "type": "uniform",
        "low": col.low,
        "high": col.high,
        "expected_mean": round((col.low + col.high) / 2, 4),
    }


def _stats_lognormal(col: LognormalColumnSpec) -> dict:
    """Extract stats from a lognormal column spec.

    Args:
        col: A validated LognormalColumnSpec.

    Returns:
        Dictionary with type, mu, sigma, expected median,
        expected mean, and optional min/max.
    """
    stats = {
        "type": "lognormal",
        "mu": col.mu,
        "sigma": col.sigma,
        "expected_median": round(np.exp(col.mu), 4),
        "expected_mean": round(np.exp(col.mu + col.sigma**2 / 2), 4),
    }
    if col.min is not None:
        stats["min"] = col.min
    if col.max is not None:
        stats["max"] = col.max
    return stats


def _stats_categorical(col: CategoricalColumnSpec) -> dict:
    """Extract stats from a categorical column spec.

    Args:
        col: A validated CategoricalColumnSpec.

    Returns:
        Dictionary with type, number of categories, values,
        and weights or "uniform" distribution indicator.
    """
    stats = {
        "type": "categorical",
        "num_categories": len(col.values),
        "values": col.values,
    }
    if col.weights is not None:
        stats["weights"] = col.weights
    else:
        stats["distribution"] = "uniform"
    return stats


def _stats_mixture(col: MixtureColumnSpec) -> dict:
    """Extract stats from a mixture column spec.

    Args:
        col: A validated MixtureColumnSpec.

    Returns:
        Dictionary with type, number of components, component
        details, and weighted expected mean.
    """
    components = []
    for comp in col.components:
        info = {"type": comp.type, "weight": comp.weight}
        if comp.type == "normal":
            info["mean"] = comp.mean
            info["std"] = comp.std
        elif comp.type == "uniform":
            info["low"] = comp.low
            info["high"] = comp.high
        components.append(info)

    expected_mean = 0.0
    for comp in col.components:
        if comp.type == "normal":
            expected_mean += comp.weight * comp.mean
        elif comp.type == "uniform":
            expected_mean += comp.weight * (comp.low + comp.high) / 2

    return {
        "type": "mixture",
        "num_components": len(col.components),
        "components": components,
        "expected_mean": round(expected_mean, 4),
    }


def _stats_derived(col: DerivedColumnSpec) -> dict:
    """Extract stats from a derived column spec.

    Args:
        col: A validated DerivedColumnSpec.

    Returns:
        Dictionary with type, expression, and dependencies.
    """
    return {
        "type": "derived",
        "expr": col.expr,
        "dependencies": col.dependencies,
    }


def _stats_conditional(col: ConditionalColumnSpec) -> dict:
    """Extract stats from a conditional column spec.

    Args:
        col: A validated ConditionalColumnSpec.

    Returns:
        Dictionary with type, condition column, and case details.
    """
    cases_summary = {}
    for case_value, case_spec in col.cases.items():
        info = {"type": case_spec.type}
        if case_spec.type == "normal":
            info["mean"] = case_spec.mean
            info["std"] = case_spec.std
        elif case_spec.type == "uniform":
            info["low"] = case_spec.low
            info["high"] = case_spec.high
        elif case_spec.type == "categorical":
            info["values"] = case_spec.values
        cases_summary[case_value] = info

    return {
        "type": "conditional",
        "condition_column": col.condition_column,
        "cases": cases_summary,
    }


_STATS_MAP = {
    NormalColumnSpec: _stats_normal,
    UniformColumnSpec: _stats_uniform,
    CategoricalColumnSpec: _stats_categorical,
    MixtureColumnSpec: _stats_mixture,
    DerivedColumnSpec: _stats_derived,
    ConditionalColumnSpec: _stats_conditional,
    LognormalColumnSpec: _stats_lognormal,
    BetaColumnSpec: _stats_beta,
    GammaColumnSpec: _stats_gamma,
    ExponentialColumnSpec: _stats_exponential,
    ParetoColumnSpec: _stats_pareto,
    StudentTColumnSpec: _stats_student_t,
    PoissonColumnSpec: _stats_poisson,
    BinomialColumnSpec: _stats_binomial,
}


def stats_only(spec: dict | TableSpec) -> dict:
    """Get statistical summary of a spec without generating data.

    Analyzes the spec and returns expected statistics for each
    column: mean, std, ranges for numeric columns; categories
    and weights for categorical; component details for mixtures;
    expressions for derived columns.

    Useful for verifying a spec before running a large generation.

    Args:
        spec: Table specification as a dictionary or TableSpec object.

    Returns:
        A dictionary with keys:
            - 'num_rows': configured row count.
            - 'num_columns': number of columns.
            - 'columns': dict mapping column names to their stats.

    Example:
        >>> from pygmalion import stats_only
        >>> stats = stats_only(spec)
        >>> print(stats["columns"]["price"]["mean"])
        100.0
    """
    if isinstance(spec, dict):
        spec = TableSpec(**spec)

    columns_stats = {}
    for col in spec.columns:
        handler = _STATS_MAP.get(type(col))
        if handler is not None:
            columns_stats[col.name] = handler(col)

    return {
        "num_rows": spec.num_rows,
        "num_columns": len(spec.columns),
        "columns": columns_stats,
    }
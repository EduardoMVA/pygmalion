"""Automatic distribution fitting for numeric data.

Fits multiple candidate distributions to a data sample,
evaluates them using AIC and the Kolmogorov-Smirnov test,
and selects the best fit.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _classify_data(series: pd.Series) -> str:
    """Classify numeric data to determine candidate distributions.

    Args:
        series: A numeric pandas Series (no NaN values).

    Returns:
        One of: "positive", "unit_interval", "non_negative_integer",
        or "real".
    """
    values = series.values

    is_integer_dtype = series.dtype in (np.int64, np.int32, np.int16, np.int8)
    is_integer_values = np.all(values == np.floor(values))
    all_non_negative = np.all(values >= 0)
    all_positive = np.all(values > 0)
    in_unit = all_positive and np.all(values <= 1)

    if is_integer_dtype and is_integer_values and all_non_negative:
        return "non_negative_integer"
    if in_unit:
        return "unit_interval"
    if all_positive:
        return "positive"
    return "real"


_CANDIDATES = {
    "real": ["normal", "student_t"],
    "positive": ["normal", "lognormal", "gamma", "exponential", "pareto"],
    "unit_interval": ["beta", "normal"],
    "non_negative_integer": ["poisson", "binomial"],
}


def _compute_aic(log_likelihood: float, num_params: int) -> float:
    """Compute Akaike Information Criterion.

    Args:
        log_likelihood: Sum of log-pdf values.
        num_params: Number of fitted parameters.

    Returns:
        AIC value. Lower is better.
    """
    return 2 * num_params - 2 * log_likelihood


def _fit_candidate(name: str, data: np.ndarray) -> dict | None:
    """Fit a single distribution candidate and evaluate it.

    Args:
        name: Distribution name (e.g., "normal", "gamma").
        data: Numpy array of observed values.

    Returns:
        Dictionary with name, params, aic, ks_statistic, ks_pvalue,
        and spec. Returns None if fitting fails.
    """
    try:
        if name == "normal":
            loc, scale = scipy_stats.norm.fit(data)
            params = (loc, scale)
            dist = scipy_stats.norm(loc=loc, scale=scale)
            spec = {"type": "normal", "mean": round(float(loc), 4), "std": round(float(scale), 4)}
            num_params = 2

        elif name == "lognormal":
            s, loc, scale = scipy_stats.lognorm.fit(data, floc=0)
            params = (s, 0, scale)
            dist = scipy_stats.lognorm(s=s, loc=0, scale=scale)
            spec = {"type": "lognormal", "mu": round(float(np.log(scale)), 4), "sigma": round(float(s), 4)}
            num_params = 2

        elif name == "gamma":
            a, loc, scale = scipy_stats.gamma.fit(data, floc=0)
            params = (a, 0, scale)
            dist = scipy_stats.gamma(a=a, loc=0, scale=scale)
            spec = {"type": "gamma", "shape": round(float(a), 4), "scale": round(float(scale), 4)}
            num_params = 2

        elif name == "exponential":
            loc, scale = scipy_stats.expon.fit(data, floc=0)
            params = (0, scale)
            dist = scipy_stats.expon(loc=0, scale=scale)
            spec = {"type": "exponential", "scale": round(float(scale), 4)}
            num_params = 1

        elif name == "pareto":
            b, loc, scale = scipy_stats.pareto.fit(data, floc=0)
            params = (b, 0, scale)
            dist = scipy_stats.pareto(b=b, loc=0, scale=scale)
            spec = {"type": "pareto", "alpha": round(float(b), 4), "scale": round(float(scale), 4)}
            num_params = 2

        elif name == "student_t":
            df, loc, scale = scipy_stats.t.fit(data)
            params = (df, loc, scale)
            dist = scipy_stats.t(df=df, loc=loc, scale=scale)
            spec = {"type": "student_t", "df": round(float(df), 4), "loc": round(float(loc), 4), "scale": round(float(scale), 4)}
            num_params = 3

        elif name == "beta":
            a, b, loc, scale = scipy_stats.beta.fit(data, floc=0, fscale=1)
            params = (a, b, 0, 1)
            dist = scipy_stats.beta(a=a, b=b, loc=0, scale=1)
            spec = {"type": "beta", "alpha": round(float(a), 4), "beta_param": round(float(b), 4)}
            num_params = 2

        elif name == "poisson":
            mu = float(np.mean(data))
            dist = scipy_stats.poisson(mu=mu)
            spec = {"type": "poisson", "mu": round(mu, 4)}
            num_params = 1
            log_ll = scipy_stats.poisson.logpmf(data, mu=mu).sum()
            ks_stat, ks_pval = scipy_stats.ks_1samp(data, scipy_stats.poisson(mu=mu).cdf)
            return {
                "name": name,
                "spec": spec,
                "aic": round(float(_compute_aic(log_ll, num_params)), 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 4),
            }

        elif name == "binomial":
            n_est = int(np.max(data))
            if n_est == 0:
                return None
            p_est = float(np.mean(data)) / n_est
            p_est = np.clip(p_est, 0.01, 0.99)
            dist = scipy_stats.binom(n=n_est, p=p_est)
            spec = {"type": "binomial", "n": n_est, "p": round(float(p_est), 4)}
            num_params = 2
            log_ll = scipy_stats.binom.logpmf(data.astype(int), n=n_est, p=p_est).sum()
            ks_stat, ks_pval = scipy_stats.ks_1samp(data, scipy_stats.binom(n=n_est, p=p_est).cdf)
            return {
                "name": name,
                "spec": spec,
                "aic": round(float(_compute_aic(log_ll, num_params)), 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 4),
            }

        else:
            return None

        log_ll = dist.logpdf(data).sum()
        if not np.isfinite(log_ll):
            return None

        ks_stat, ks_pval = scipy_stats.kstest(data, dist.cdf)

        return {
            "name": name,
            "spec": spec,
            "aic": round(float(_compute_aic(log_ll, num_params)), 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 4),
        }

    except Exception:
        return None


def fit_best_distribution(
    series: pd.Series,
    ks_threshold: float = 0.05,
) -> dict:
    """Find the best-fitting distribution for a numeric series.

    Tries multiple candidate distributions based on data
    characteristics, evaluates each with AIC and KS test,
    and returns the best fit.

    If no parametric distribution passes the KS test,
    falls back to bootstrap.

    Args:
        series: A numeric pandas Series (NaN values will be dropped).
        ks_threshold: p-value threshold for KS test. Distributions
            with p-value below this are rejected. Defaults to 0.05.

    Returns:
        A column spec dictionary ready to use in a TableSpec.
        Includes the fitted distribution type and parameters.

    Example:
        >>> import pandas as pd
        >>> data = pd.Series(np.random.lognormal(10, 0.5, 1000))
        >>> spec = fit_best_distribution(data)
        >>> print(spec["type"])
        'lognormal'
    """
    series = series.dropna()
    data = series.values.astype(float)

    if len(data) < 5:
        return {
            "name": series.name,
            "type": "bootstrap",
            "values": series.tolist(),
        }

    data_class = _classify_data(series)
    candidates = _CANDIDATES.get(data_class, ["normal"])

    results = []
    for name in candidates:
        result = _fit_candidate(name, data)
        if result is not None:
            results.append(result)

    passed_ks = [r for r in results if r["ks_pvalue"] >= ks_threshold]

    if passed_ks:
        best = min(passed_ks, key=lambda r: r["aic"])
    elif results:
        best = min(results, key=lambda r: r["aic"])
    else:
        return {
            "name": series.name,
            "type": "bootstrap",
            "values": series.tolist(),
        }

    spec = best["spec"]
    spec["name"] = series.name

    if data_class in ("positive", "real"):
        spec["min"] = round(float(np.min(data)), 4)
        spec["max"] = round(float(np.max(data)), 4)

    return spec
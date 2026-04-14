import numpy as np
import pandas as pd
import pytest

from pygmalion.io.fitting import fit_best_distribution, _classify_data


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_classify_positive(rng):
    data = pd.Series(rng.lognormal(5, 0.5, 1000))
    assert _classify_data(data) == "positive"


def test_classify_real(rng):
    data = pd.Series(rng.normal(0, 1, 1000))
    assert _classify_data(data) == "real"


def test_classify_unit_interval(rng):
    data = pd.Series(rng.beta(2, 5, 1000))
    assert _classify_data(data) == "unit_interval"


def test_classify_integer(rng):
    data = pd.Series(rng.poisson(5, 1000))
    assert _classify_data(data) == "non_negative_integer"


def test_fit_normal_data(rng):
    data = pd.Series(rng.normal(100, 10, 2000), name="x")
    spec = fit_best_distribution(data)
    assert spec["type"] in ("normal", "student_t")
    assert spec["name"] == "x"


def test_fit_lognormal_data(rng):
    data = pd.Series(rng.lognormal(10, 0.5, 2000), name="salary")
    spec = fit_best_distribution(data)
    assert spec["type"] in ("lognormal", "gamma")
    assert spec["name"] == "salary"


def test_fit_poisson_data(rng):
    data = pd.Series(rng.poisson(7, 2000), name="calls")
    spec = fit_best_distribution(data)
    assert spec["type"] in ("poisson", "binomial")
    assert spec["name"] == "calls"


def test_fit_beta_data(rng):
    data = pd.Series(rng.beta(2, 5, 2000), name="rate")
    spec = fit_best_distribution(data)
    assert spec["type"] == "beta"


def test_fit_few_data_falls_back_to_bootstrap():
    data = pd.Series([1, 2, 3], name="x")
    spec = fit_best_distribution(data)
    assert spec["type"] == "bootstrap"


def test_fit_result_is_usable(rng):
    from pygmalion.engine.synthesizer import synthesize

    data = pd.Series(rng.normal(50, 10, 1000), name="x")
    col_spec = fit_best_distribution(data)
    table_spec = {
        "num_rows": 100,
        "columns": [col_spec],
    }
    df = synthesize(table_spec)
    assert len(df) == 100
import pandas as pd
import polars as pl

from pygmalion.engine.synthesizer import synthesize


def test_synthesize_una_columna_normal():
    spec = {
        "num_rows": 100,
        "columns": [
            {"name": "precio", "type": "normal", "mean": 100, "std": 10}
        ],
    }
    df = synthesize(spec)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "precio" in df.columns


def test_synthesize_output_polars():
    spec = {
        "num_rows": 50,
        "columns": [
            {"name": "x", "type": "uniform", "low": 0, "high": 1}
        ],
    }
    df = synthesize(spec, output_format="polars")
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 50


def test_synthesize_columnas_mixtas():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "precio", "type": "normal", "mean": 100, "std": 15},
            {"name": "edad", "type": "uniform", "low": 18, "high": 65},
            {"name": "color", "type": "categorical", "values": ["rojo", "azul"]},
        ],
    }
    df = synthesize(spec)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert list(df.columns) == ["precio", "edad", "color"]


def test_synthesize_con_mixture():
    spec = {
        "num_rows": 1000,
        "columns": [
            {
                "name": "salario",
                "type": "mixture",
                "components": [
                    {"type": "normal", "mean": 30000, "std": 5000, "weight": 0.7},
                    {"type": "normal", "mean": 80000, "std": 15000, "weight": 0.3},
                ],
            }
        ],
    }
    df = synthesize(spec)
    assert len(df) == 1000
    assert "salario" in df.columns


def test_synthesize_output_format_invalido():
    import pytest

    spec = {
        "num_rows": 10,
        "columns": [
            {"name": "x", "type": "uniform", "low": 0, "high": 1}
        ],
    }
    with pytest.raises(ValueError):
        synthesize(spec, output_format="excel")


def test_synthesize_desde_tablespec():
    from pygmalion.schema.spec import TableSpec, NormalColumnSpec

    table = TableSpec(
        num_rows=50,
        columns=[NormalColumnSpec(name="y", type="normal", mean=0, std=1)],
    )
    df = synthesize(table)
    assert len(df) == 50
    assert "y" in df.columns


def test_synthesize_columna_derivada():
    spec = {
        "num_rows": 100,
        "columns": [
            {"name": "precio", "type": "uniform", "low": 10, "high": 100},
            {"name": "cantidad", "type": "uniform", "low": 1, "high": 5},
            {
                "name": "total",
                "type": "derived",
                "expr": "precio * cantidad",
                "dependencies": ["precio", "cantidad"],
            },
        ],
    }
    df = synthesize(spec)
    assert "total" in df.columns
    assert len(df) == 100
    assert all(df["total"] == df["precio"] * df["cantidad"])


def test_synthesize_derivada_de_derivada():
    spec = {
        "num_rows": 50,
        "columns": [
            {"name": "precio", "type": "uniform", "low": 10, "high": 100},
            {"name": "cantidad", "type": "uniform", "low": 1, "high": 5},
            {
                "name": "subtotal",
                "type": "derived",
                "expr": "precio * cantidad",
                "dependencies": ["precio", "cantidad"],
            },
            {
                "name": "total",
                "type": "derived",
                "expr": "subtotal * 1.16",
                "dependencies": ["subtotal"],
            },
        ],
    }
    df = synthesize(spec)
    assert "total" in df.columns
    expected = df["precio"] * df["cantidad"] * 1.16
    assert all(abs(df["total"] - expected) < 1e-6)


def test_synthesize_dependencia_inexistente():
    import pytest

    spec = {
        "num_rows": 10,
        "columns": [
            {
                "name": "total",
                "type": "derived",
                "expr": "precio * 2",
                "dependencies": ["precio"],
            },
        ],
    }
    with pytest.raises(ValueError):
        synthesize(spec)


def test_synthesize_orden_columnas_respetado():
    spec = {
        "num_rows": 50,
        "columns": [
            {"name": "a", "type": "uniform", "low": 0, "high": 1},
            {
                "name": "b",
                "type": "derived",
                "expr": "a * 2",
                "dependencies": ["a"],
            },
            {"name": "c", "type": "normal", "mean": 0, "std": 1},
        ],
    }
    df = synthesize(spec)
    assert list(df.columns) == ["a", "b", "c"]

def test_synthesize_conditional():
    spec = {
        "num_rows": 1000,
        "columns": [
            {
                "name": "nivel",
                "type": "categorical",
                "values": ["junior", "senior"],
                "weights": [0.6, 0.4],
            },
            {
                "name": "salario",
                "type": "conditional",
                "condition_column": "nivel",
                "cases": {
                    "junior": {"type": "normal", "mean": 25000, "std": 3000},
                    "senior": {"type": "normal", "mean": 60000, "std": 10000},
                },
            },
        ],
    }
    df = synthesize(spec)
    assert len(df) == 1000
    assert "salario" in df.columns

    juniors = df[df["nivel"] == "junior"]["salario"]
    seniors = df[df["nivel"] == "senior"]["salario"]
    assert abs(juniors.mean() - 25000) < 2000
    assert abs(seniors.mean() - 60000) < 5000


def test_synthesize_conditional_dependencia_faltante():
    import pytest

    spec = {
        "num_rows": 100,
        "columns": [
            {
                "name": "salario",
                "type": "conditional",
                "condition_column": "nivel",
                "cases": {
                    "junior": {"type": "normal", "mean": 25000, "std": 3000},
                },
            },
        ],
    }
    with pytest.raises(ValueError):
        synthesize(spec)


def test_synthesize_con_constraint():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "edad", "type": "uniform", "low": 18, "high": 65},
            {"name": "experiencia", "type": "uniform", "low": 0, "high": 30},
        ],
        "constraints": ["experiencia <= edad - 18"],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert all(df["experiencia"] <= df["edad"] - 18)


def test_synthesize_constraint_imposible():
    import pytest

    spec = {
        "num_rows": 100,
        "columns": [
            {"name": "x", "type": "uniform", "low": 0, "high": 10},
        ],
        "constraints": ["x > 100"],
    }
    with pytest.raises(RuntimeError):
        synthesize(spec, max_attempts=3)


def test_synthesize_bootstrap():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "precio", "type": "bootstrap", "values": [100, 200, 300, 400, 500]},
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert set(df["precio"].unique()).issubset({100, 200, 300, 400, 500})

def test_synthesize_lognormal():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "salario", "type": "lognormal", "mu": 10.5, "sigma": 0.5}
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert all(df["salario"] > 0)

def test_synthesize_beta():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "tasa", "type": "beta", "alpha": 2, "beta_param": 5}
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert all(df["tasa"] >= 0)
    assert all(df["tasa"] <= 1)

def test_synthesize_gamma():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "tiempo", "type": "gamma", "shape": 2, "scale": 10}
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert all(df["tiempo"] > 0)

def test_synthesize_exponential():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "tiempo", "type": "exponential", "scale": 5}
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert all(df["tiempo"] > 0)

def test_synthesize_pareto():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "ingreso", "type": "pareto", "alpha": 2.5, "scale": 1000}
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert all(df["ingreso"] >= 1000)

def test_synthesize_student_t():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "retorno", "type": "student_t", "df": 5, "loc": 0, "scale": 1}
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500

def test_synthesize_poisson():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "llamadas", "type": "poisson", "mu": 5}
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert all(df["llamadas"] >= 0)


def test_synthesize_binomial():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "aprobados", "type": "binomial", "n": 20, "p": 0.8}
        ],
    }
    df = synthesize(spec)
    assert len(df) == 500
    assert all(df["aprobados"] >= 0)
    assert all(df["aprobados"] <= 20)

def test_synthesize_seed_reproducible():
    spec = {
        "num_rows": 100,
        "columns": [
            {"name": "x", "type": "normal", "mean": 0, "std": 1}
        ],
        "seed": 42,
    }
    df1 = synthesize(spec)
    df2 = synthesize(spec)
    assert list(df1["x"]) == list(df2["x"])


def test_synthesize_seed_diferente_resultado_diferente():
    spec1 = {
        "num_rows": 100,
        "columns": [
            {"name": "x", "type": "normal", "mean": 0, "std": 1}
        ],
        "seed": 42,
    }
    spec2 = {
        "num_rows": 100,
        "columns": [
            {"name": "x", "type": "normal", "mean": 0, "std": 1}
        ],
        "seed": 99,
    }
    df1 = synthesize(spec1)
    df2 = synthesize(spec2)
    assert list(df1["x"]) != list(df2["x"])


def test_synthesize_sin_seed_varia():
    spec = {
        "num_rows": 100,
        "columns": [
            {"name": "x", "type": "normal", "mean": 0, "std": 1}
        ],
    }
    df1 = synthesize(spec)
    df2 = synthesize(spec)
    assert list(df1["x"]) != list(df2["x"])
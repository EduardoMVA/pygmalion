from pathlib import Path

import pandas as pd
import pytest

from pygmalion.io.reader import learn_from_csv


@pytest.fixture
def csv_mixto(tmp_path):
    df = pd.DataFrame({
        "precio": [100.0, 200.0, 150.0, 300.0, 250.0,
                    180.0, 220.0, 170.0, 190.0, 210.0,
                    160.0, 240.0],
        "categoria": ["A", "B", "A", "C", "B",
                       "A", "C", "B", "A", "B",
                       "C", "A"],
        "habitaciones": [1, 2, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3],
    })
    path = tmp_path / "datos.csv"
    df.to_csv(path, index=False)
    return path


def test_learn_devuelve_spec_valido(csv_mixto):
    spec = learn_from_csv(csv_mixto)
    assert "num_rows" in spec
    assert "columns" in spec
    assert spec["num_rows"] == 12


def test_learn_detecta_tipos(csv_mixto):
    spec = learn_from_csv(csv_mixto)
    cols = {c["name"]: c["type"] for c in spec["columns"]}
    assert cols["precio"] == "normal"
    assert cols["categoria"] == "categorical"
    assert cols["habitaciones"] == "categorical"


def test_learn_parametros_numericos(csv_mixto):
    spec = learn_from_csv(csv_mixto)
    precio = next(c for c in spec["columns"] if c["name"] == "precio")
    assert "mean" in precio
    assert "std" in precio
    assert "min" in precio
    assert "max" in precio
    assert precio["std"] > 0


def test_learn_parametros_categoricos(csv_mixto):
    spec = learn_from_csv(csv_mixto)
    cat = next(c for c in spec["columns"] if c["name"] == "categoria")
    assert "values" in cat
    assert "weights" in cat
    assert abs(sum(cat["weights"]) - 1.0) < 1e-6


def test_learn_num_rows_personalizado(csv_mixto):
    spec = learn_from_csv(csv_mixto, num_rows=5000)
    assert spec["num_rows"] == 5000


def test_learn_spec_es_usable(csv_mixto):
    from pygmalion.engine.synthesizer import synthesize

    spec = learn_from_csv(csv_mixto, num_rows=100)
    df = synthesize(spec)
    assert len(df) == 100


from pygmalion.io.reader import template_from_data


def test_template_devuelve_spec(csv_mixto):
    spec = template_from_data(csv_mixto)
    assert "num_rows" in spec
    assert "columns" in spec
    assert spec["num_rows"] == 1000


def test_template_num_rows_custom(csv_mixto):
    spec = template_from_data(csv_mixto, num_rows=500)
    assert spec["num_rows"] == 500


def test_template_numerico_redondeado(csv_mixto):
    spec = template_from_data(csv_mixto)
    precio = next(c for c in spec["columns"] if c["name"] == "precio")
    assert isinstance(precio["mean"], int) or precio["mean"] == int(precio["mean"])


def test_template_categorico_sin_weights(csv_mixto):
    spec = template_from_data(csv_mixto)
    cat = next(c for c in spec["columns"] if c["name"] == "categoria")
    assert "values" in cat
    assert "weights" not in cat
    assert cat["values"] == sorted(cat["values"])


def test_template_es_usable(csv_mixto):
    from pygmalion.engine.synthesizer import synthesize

    spec = template_from_data(csv_mixto, num_rows=50)
    df = synthesize(spec)
    assert len(df) == 50

def test_learn_bootstrap(csv_mixto):
    spec = learn_from_csv(csv_mixto, strategy="bootstrap")
    assert all(c["type"] == "bootstrap" for c in spec["columns"])


def test_learn_bootstrap_valores_presentes(csv_mixto):
    spec = learn_from_csv(csv_mixto, strategy="bootstrap")
    precio_col = next(c for c in spec["columns"] if c["name"] == "precio")
    assert "values" in precio_col
    assert len(precio_col["values"]) > 0


def test_learn_bootstrap_es_usable(csv_mixto):
    from pygmalion.engine.synthesizer import synthesize

    spec = learn_from_csv(csv_mixto, num_rows=200, strategy="bootstrap")
    df = synthesize(spec)
    assert len(df) == 200


def test_learn_strategy_invalida(csv_mixto):
    import pytest
    with pytest.raises(ValueError):
        learn_from_csv(csv_mixto, strategy="inventada")

def test_learn_auto_fit(csv_mixto):
    spec = learn_from_csv(csv_mixto, strategy="auto_fit")
    assert "num_rows" in spec
    assert "columns" in spec
    col_types = {c["name"]: c["type"] for c in spec["columns"]}
    assert col_types["categoria"] == "categorical"
    assert col_types["precio"] in ("normal", "lognormal", "gamma", "uniform", "bootstrap")


def test_learn_auto_fit_es_usable(csv_mixto):
    from pygmalion.engine.synthesizer import synthesize

    spec = learn_from_csv(csv_mixto, num_rows=100, strategy="auto_fit")
    df = synthesize(spec)
    assert len(df) == 100


def test_learn_auto_fit_con_datos_grandes(tmp_path):
    import numpy as np
    from pygmalion.engine.synthesizer import synthesize

    df = pd.DataFrame({
        "salario": np.random.lognormal(10, 0.5, 500),
        "edad": np.random.normal(35, 8, 500),
        "depto": np.random.choice(["A", "B", "C"], 500),
    })
    path = tmp_path / "grande.csv"
    df.to_csv(path, index=False)

    spec = learn_from_csv(path, num_rows=200, strategy="auto_fit")
    col_types = {c["name"]: c["type"] for c in spec["columns"]}
    assert col_types["depto"] == "categorical"
    assert col_types["salario"] in ("lognormal", "gamma", "normal", "bootstrap")
    assert col_types["edad"] in ("normal", "student_t", "bootstrap")

    df_synth = synthesize(spec)
    assert len(df_synth) == 200
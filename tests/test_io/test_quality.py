import pandas as pd
import numpy as np

from pygmalion.io.quality import quality_report


def test_quality_report_basico():
    real = pd.DataFrame({
        "precio": np.random.normal(100, 10, 1000),
        "color": np.random.choice(["rojo", "azul"], 1000),
    })
    synthetic = pd.DataFrame({
        "precio": np.random.normal(100, 10, 1000),
        "color": np.random.choice(["rojo", "azul"], 1000),
    })
    report = quality_report(real, synthetic)
    assert "overall_score" in report
    assert "columns" in report
    assert report["num_columns_compared"] == 2


def test_quality_score_datos_similares():
    data = np.random.normal(50, 5, 5000)
    real = pd.DataFrame({"x": data[:2500]})
    synthetic = pd.DataFrame({"x": data[2500:]})
    report = quality_report(real, synthetic)
    assert report["columns"]["x"]["score"] > 0.9


def test_quality_score_datos_diferentes():
    real = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
    synthetic = pd.DataFrame({"x": np.random.normal(100, 1, 1000)})
    report = quality_report(real, synthetic)
    assert report["columns"]["x"]["score"] < 0.5


def test_quality_categoricas():
    real = pd.DataFrame({"color": ["rojo"] * 700 + ["azul"] * 300})
    synthetic = pd.DataFrame({"color": ["rojo"] * 680 + ["azul"] * 320})
    report = quality_report(real, synthetic)
    assert report["columns"]["color"]["type"] == "categorical"
    assert report["columns"]["color"]["score"] > 0.9


def test_quality_sin_columnas_comunes():
    import pytest

    real = pd.DataFrame({"a": [1, 2, 3]})
    synthetic = pd.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(ValueError):
        quality_report(real, synthetic)


def test_quality_report_integrado():
    from pygmalion.engine.synthesizer import synthesize
    from pygmalion.io.reader import learn_from_csv
    import tempfile
    from pathlib import Path

    real = pd.DataFrame({
        "precio": np.random.normal(100, 10, 500),
        "cantidad": np.random.uniform(1, 10, 500),
        "tipo": np.random.choice(["A", "B", "C"], 500),
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        real.to_csv(f, index=False)
        csv_path = f.name

    spec = learn_from_csv(csv_path, num_rows=500)
    synthetic = synthesize(spec)

    report = quality_report(real, synthetic)
    assert report["overall_score"] > 0.5
    assert report["num_columns_compared"] == 3
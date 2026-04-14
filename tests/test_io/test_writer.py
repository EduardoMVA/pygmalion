import json
from pathlib import Path

import polars as pl
import pandas as pd

from pygmalion.io.writer import to_csv, to_json


def test_to_csv_polars(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "test.csv"
    to_csv(df, path)
    assert path.exists()
    content = path.read_text()
    assert "a,b" in content
    assert "1,x" in content


def test_to_csv_pandas(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "test.csv"
    to_csv(df, path)
    assert path.exists()


def test_to_json_polars(tmp_path):
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "test.json"
    to_json(df, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 2
    assert data[0]["a"] == 1
    assert data[0]["b"] == "x"


def test_to_json_pandas(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "test.json"
    to_json(df, path)
    assert path.exists()


def test_to_json_caracteres_especiales(tmp_path):
    df = pl.DataFrame({"nombre": ["José", "María", "Señor"]})
    path = tmp_path / "test.json"
    to_json(df, path)
    content = path.read_text(encoding="utf-8")
    assert "José" in content
    assert "María" in content
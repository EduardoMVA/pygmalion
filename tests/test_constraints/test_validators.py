import numpy as np
import polars as pl

from pygmalion.constraints.validators import apply_constraints


def test_apply_sin_constraints():
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = apply_constraints(df, [])
    assert len(result) == 3


def test_apply_filtra_correctamente():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = apply_constraints(df, ["a > 2"])
    assert len(result) == 3
    assert result["a"].to_list() == [3, 4, 5]


def test_apply_multiples_constraints():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    result = apply_constraints(df, ["a > 1", "b > 1"])
    assert len(result) == 3
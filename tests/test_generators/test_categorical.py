import numpy as np
from pygmalion.generators.categorical import CategoricalGenerator
from pygmalion.schema.spec import CategoricalColumnSpec
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def test_categorical_genera_n_valores(rng):
    spec = CategoricalColumnSpec(
        name="color", type="categorical", values=["rojo", "azul", "verde"]
    )
    gen = CategoricalGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_categorical_valores_validos(rng):
    spec = CategoricalColumnSpec(
        name="color", type="categorical", values=["rojo", "azul"]
    )
    gen = CategoricalGenerator(spec)
    result = gen.generate(1000, rng)
    assert set(result).issubset({"rojo", "azul"})


def test_categorical_con_weights(rng):
    spec = CategoricalColumnSpec(
        name="color",
        type="categorical",
        values=["rojo", "azul"],
        weights=[0.9, 0.1],
    )
    gen = CategoricalGenerator(spec)
    result = gen.generate(10000, rng)
    freq_rojo = np.mean(result == "rojo")
    assert abs(freq_rojo - 0.9) < 0.05


def test_categorical_sin_weights_distribucion_uniforme(rng):
    spec = CategoricalColumnSpec(
        name="letra", type="categorical", values=["A", "B", "C", "D"]
    )
    gen = CategoricalGenerator(spec)
    result = gen.generate(10000, rng)
    freq_a = np.mean(result == "A")
    assert abs(freq_a - 0.25) < 0.05
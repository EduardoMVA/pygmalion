import numpy as np

from pygmalion.generators.bootstrap import BootstrapGenerator
from pygmalion.schema.spec import BootstrapColumnSpec

import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def test_bootstrap_genera_n_valores(rng):
    spec = BootstrapColumnSpec(
        name="x", type="bootstrap", values=[10, 20, 30, 40, 50]
    )
    gen = BootstrapGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_bootstrap_solo_valores_originales(rng):
    spec = BootstrapColumnSpec(
        name="x", type="bootstrap", values=[1, 2, 3]
    )
    gen = BootstrapGenerator(spec)
    result = gen.generate(5000, rng)
    assert set(result).issubset({1, 2, 3})


def test_bootstrap_frecuencias_proporcionales(rng):
    values = [1] * 80 + [2] * 20
    spec = BootstrapColumnSpec(
        name="x", type="bootstrap", values=values
    )
    gen = BootstrapGenerator(spec)
    result = gen.generate(10000, rng)
    freq_1 = np.mean(result == 1)
    assert abs(freq_1 - 0.8) < 0.05


def test_bootstrap_strings(rng):
    spec = BootstrapColumnSpec(
        name="ciudad", type="bootstrap",
        values=["CDMX", "Monterrey", "Guadalajara"]
    )
    gen = BootstrapGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000
    assert set(result).issubset({"CDMX", "Monterrey", "Guadalajara"})


def test_bootstrap_un_solo_valor(rng):
    spec = BootstrapColumnSpec(
        name="x", type="bootstrap", values=[42]
    )
    gen = BootstrapGenerator(spec)
    result = gen.generate(100, rng)
    assert all(result == 42)
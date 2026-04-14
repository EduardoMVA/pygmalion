import numpy as np
from pygmalion.generators.mixture import MixtureGenerator
from pygmalion.schema.spec import MixtureColumnSpec
import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def test_mixture_genera_n_valores(rng):
    spec = MixtureColumnSpec(
        name="salario",
        type="mixture",
        components=[
            {"type": "normal", "mean": 30000, "std": 5000, "weight": 0.7},
            {"type": "normal", "mean": 80000, "std": 15000, "weight": 0.3},
        ],
    )
    gen = MixtureGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_mixture_media_ponderada(rng):
    spec = MixtureColumnSpec(
        name="valor",
        type="mixture",
        components=[
            {"type": "normal", "mean": 0, "std": 1, "weight": 0.5},
            {"type": "normal", "mean": 100, "std": 1, "weight": 0.5},
        ],
    )
    gen = MixtureGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 50) < 5


def test_mixture_componentes_mixtos(rng):
    spec = MixtureColumnSpec(
        name="valor",
        type="mixture",
        components=[
            {"type": "normal", "mean": 50, "std": 5, "weight": 0.6},
            {"type": "uniform", "low": 0, "high": 100, "weight": 0.4},
        ],
    )
    gen = MixtureGenerator(spec)
    result = gen.generate(10000, rng)
    assert len(result) == 10000


def test_mixture_shuffle(rng):
    spec = MixtureColumnSpec(
        name="valor",
        type="mixture",
        components=[
            {"type": "normal", "mean": 0, "std": 1, "weight": 0.5},
            {"type": "normal", "mean": 1000, "std": 1, "weight": 0.5},
        ],
    )
    gen = MixtureGenerator(spec)
    result = gen.generate(1000, rng)
    first_half = result[:500]
    has_low = np.any(first_half < 100)
    has_high = np.any(first_half > 900)
    assert has_low and has_high
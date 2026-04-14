import pytest
import numpy as np
from pygmalion.generators.base import BaseGenerator
from pygmalion.generators.registry import (
    register,
    get_generator,
    list_registered,
    clear_registry,
)

@pytest.fixture
def rng():
    return np.random.default_rng(42)

class FakeGenerator(BaseGenerator):
    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return np.zeros(n)


class NotAGenerator:
    pass


@pytest.fixture(autouse=True)
def limpiar_registry():
    clear_registry()
    yield
    clear_registry()
    import importlib
    import pygmalion.generators.numeric
    import pygmalion.generators.categorical
    import pygmalion.generators.mixture
    import pygmalion.generators.conditional
    import pygmalion.generators.bootstrap
    importlib.reload(pygmalion.generators.numeric)
    importlib.reload(pygmalion.generators.categorical)
    importlib.reload(pygmalion.generators.mixture)
    importlib.reload(pygmalion.generators.conditional)
    importlib.reload(pygmalion.generators.bootstrap)


def test_register_y_get():
    register("fake", FakeGenerator)
    gen_class = get_generator("fake")
    assert gen_class is FakeGenerator


def test_register_duplicado():
    register("fake", FakeGenerator)
    with pytest.raises(ValueError):
        register("fake", FakeGenerator)


def test_get_no_registrado():
    with pytest.raises(KeyError):
        get_generator("inexistente")


def test_register_clase_invalida():
    with pytest.raises(TypeError):
        register("malo", NotAGenerator)


def test_list_registered():
    register("fake", FakeGenerator)
    assert "fake" in list_registered()


def test_generate_funciona(rng):
    register("fake", FakeGenerator)
    gen_class = get_generator("fake")
    gen = gen_class()
    rng = np.random.default_rng(42)
    result = gen.generate(10, rng)
    assert len(result) == 10
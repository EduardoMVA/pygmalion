import numpy as np
from pygmalion.generators.numeric import (
    NormalGenerator, UniformGenerator, LognormalGenerator,
    BetaGenerator, GammaGenerator, ExponentialGenerator,
    ParetoGenerator, StudentTGenerator,
    PoissonGenerator, BinomialGenerator,
)
from pygmalion.schema.spec import (
    NormalColumnSpec, UniformColumnSpec, LognormalColumnSpec,
    BetaColumnSpec, GammaColumnSpec, ExponentialColumnSpec,
    ParetoColumnSpec, StudentTColumnSpec,
    PoissonColumnSpec, BinomialColumnSpec,
)

import pytest

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def test_normal_genera_n_valores(rng):
    spec = NormalColumnSpec(name="x", type="normal", mean=0, std=1)
    gen = NormalGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_normal_media_aproximada(rng):
    spec = NormalColumnSpec(name="x", type="normal", mean=100, std=10)
    gen = NormalGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 100) < 2


def test_normal_std_aproximada(rng):
    spec = NormalColumnSpec(name="x", type="normal", mean=0, std=5)
    gen = NormalGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.std(result) - 5) < 1


def test_normal_truncada_respeta_bounds(rng):
    spec = NormalColumnSpec(name="x", type="normal", mean=50, std=10, min=30, max=70)
    gen = NormalGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 30)
    assert np.all(result <= 70)


def test_normal_truncada_solo_min(rng):
    spec = NormalColumnSpec(name="x", type="normal", mean=50, std=10, min=40)
    gen = NormalGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 40)


def test_normal_truncada_solo_max(rng):
    spec = NormalColumnSpec(name="x", type="normal", mean=50, std=10, max=60)
    gen = NormalGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result <= 60)


def test_uniform_genera_n_valores(rng):
    spec = UniformColumnSpec(name="x", type="uniform", low=0, high=100)
    gen = UniformGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_uniform_respeta_rango(rng):
    spec = UniformColumnSpec(name="x", type="uniform", low=10, high=20)
    gen = UniformGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 10)
    assert np.all(result <= 20)


def test_uniform_media_aproximada(rng):
    spec = UniformColumnSpec(name="x", type="uniform", low=0, high=100)
    gen = UniformGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 50) < 3


from pygmalion.generators.numeric import NormalGenerator, UniformGenerator, LognormalGenerator
from pygmalion.schema.spec import NormalColumnSpec, UniformColumnSpec, LognormalColumnSpec


def test_lognormal_genera_n_valores(rng):
    spec = LognormalColumnSpec(name="x", type="lognormal", mu=0, sigma=1)
    gen = LognormalGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_lognormal_valores_positivos(rng):
    spec = LognormalColumnSpec(name="x", type="lognormal", mu=5, sigma=0.5)
    gen = LognormalGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result > 0)


def test_lognormal_mediana_aproximada(rng):
    spec = LognormalColumnSpec(name="x", type="lognormal", mu=10, sigma=0.3)
    gen = LognormalGenerator(spec)
    result = gen.generate(10000, rng)
    expected_median = np.exp(10)
    assert abs(np.median(result) - expected_median) / expected_median < 0.1


def test_lognormal_con_truncamiento(rng):
    spec = LognormalColumnSpec(
        name="x", type="lognormal", mu=5, sigma=0.5, min=50, max=500
    )
    gen = LognormalGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 50)
    assert np.all(result <= 500)

def test_beta_genera_n_valores(rng):
    spec = BetaColumnSpec(name="x", type="beta", alpha=2, beta_param=5)
    gen = BetaGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_beta_rango_default(rng):
    spec = BetaColumnSpec(name="x", type="beta", alpha=2, beta_param=5)
    gen = BetaGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_beta_rango_custom(rng):
    spec = BetaColumnSpec(
        name="x", type="beta", alpha=2, beta_param=5, low=10, high=100
    )
    gen = BetaGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 10)
    assert np.all(result <= 100)


def test_beta_media_aproximada(rng):
    spec = BetaColumnSpec(name="x", type="beta", alpha=2, beta_param=2)
    gen = BetaGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 0.5) < 0.05

def test_gamma_genera_n_valores(rng):
    spec = GammaColumnSpec(name="x", type="gamma", shape=2, scale=10)
    gen = GammaGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_gamma_valores_positivos(rng):
    spec = GammaColumnSpec(name="x", type="gamma", shape=2, scale=10)
    gen = GammaGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result > 0)


def test_gamma_media_aproximada(rng):
    spec = GammaColumnSpec(name="x", type="gamma", shape=3, scale=10)
    gen = GammaGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 30) < 3


def test_gamma_shape_uno_es_exponencial(rng):
    spec = GammaColumnSpec(name="x", type="gamma", shape=1, scale=5)
    gen = GammaGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 5) < 1

def test_exponential_genera_n_valores(rng):
    spec = ExponentialColumnSpec(name="x", type="exponential", scale=5)
    gen = ExponentialGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_exponential_valores_positivos(rng):
    spec = ExponentialColumnSpec(name="x", type="exponential", scale=10)
    gen = ExponentialGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result > 0)


def test_exponential_media_aproximada(rng):
    spec = ExponentialColumnSpec(name="x", type="exponential", scale=5)
    gen = ExponentialGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 5) < 1


def test_exponential_con_rate_media_aproximada(rng):
    spec = ExponentialColumnSpec(name="x", type="exponential", rate=0.1)
    gen = ExponentialGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 10) < 2

def test_pareto_genera_n_valores(rng):
    spec = ParetoColumnSpec(name="x", type="pareto", alpha=2.5, scale=100)
    gen = ParetoGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_pareto_valores_mayores_que_scale(rng):
    spec = ParetoColumnSpec(name="x", type="pareto", alpha=2, scale=500)
    gen = ParetoGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 500)


def test_pareto_media_aproximada(rng):
    spec = ParetoColumnSpec(name="x", type="pareto", alpha=3, scale=100)
    gen = ParetoGenerator(spec)
    result = gen.generate(10000, rng)
    expected_mean = 100 * 3 / (3 - 1)
    assert abs(np.mean(result) - expected_mean) < 20


def test_pareto_cola_pesada(rng):
    spec = ParetoColumnSpec(name="x", type="pareto", alpha=1.5, scale=100)
    gen = ParetoGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.max(result) > np.median(result) * 5

def test_student_t_genera_n_valores(rng):
    spec = StudentTColumnSpec(name="x", type="student_t", df=5)
    gen = StudentTGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_student_t_media_aproximada(rng):
    spec = StudentTColumnSpec(name="x", type="student_t", df=10, loc=50, scale=5)
    gen = StudentTGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 50) < 2


def test_student_t_colas_mas_pesadas_que_normal(rng):
    spec_t = StudentTColumnSpec(name="x", type="student_t", df=3, loc=0, scale=1)
    gen_t = StudentTGenerator(spec_t)
    result_t = gen_t.generate(10000, rng)

    spec_n = NormalColumnSpec(name="x", type="normal", mean=0, std=1)
    gen_n = NormalGenerator(spec_n)
    result_n = gen_n.generate(10000, rng)

    extreme_t = np.mean(np.abs(result_t) > 3)
    extreme_n = np.mean(np.abs(result_n) > 3)
    assert extreme_t > extreme_n

def test_poisson_genera_n_valores(rng):
    spec = PoissonColumnSpec(name="x", type="poisson", mu=5)
    gen = PoissonGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_poisson_valores_enteros(rng):
    spec = PoissonColumnSpec(name="x", type="poisson", mu=10)
    gen = PoissonGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result == result.astype(int))


def test_poisson_valores_no_negativos(rng):
    spec = PoissonColumnSpec(name="x", type="poisson", mu=3)
    gen = PoissonGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 0)


def test_poisson_media_aproximada(rng):
    spec = PoissonColumnSpec(name="x", type="poisson", mu=7)
    gen = PoissonGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 7) < 1


def test_binomial_genera_n_valores(rng):
    spec = BinomialColumnSpec(name="x", type="binomial", n=20, p=0.5)
    gen = BinomialGenerator(spec)
    result = gen.generate(1000, rng)
    assert len(result) == 1000


def test_binomial_valores_enteros(rng):
    spec = BinomialColumnSpec(name="x", type="binomial", n=10, p=0.5)
    gen = BinomialGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result == result.astype(int))


def test_binomial_rango_correcto(rng):
    spec = BinomialColumnSpec(name="x", type="binomial", n=15, p=0.5)
    gen = BinomialGenerator(spec)
    result = gen.generate(10000, rng)
    assert np.all(result >= 0)
    assert np.all(result <= 15)


def test_binomial_media_aproximada(rng):
    spec = BinomialColumnSpec(name="x", type="binomial", n=20, p=0.3)
    gen = BinomialGenerator(spec)
    result = gen.generate(10000, rng)
    assert abs(np.mean(result) - 6) < 1
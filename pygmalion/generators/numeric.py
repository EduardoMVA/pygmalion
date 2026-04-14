"""Generators for numeric distributions."""

import numpy as np
from scipy import stats

from pygmalion.generators.base import BaseGenerator
from pygmalion.generators.registry import register
from pygmalion.schema.spec import (
    NormalColumnSpec, UniformColumnSpec, LognormalColumnSpec,
    BetaColumnSpec, GammaColumnSpec, ExponentialColumnSpec,
    ParetoColumnSpec, StudentTColumnSpec,
    PoissonColumnSpec, BinomialColumnSpec,
)


class NormalGenerator(BaseGenerator):
    """Generator for normally distributed values.

    Supports optional truncation via min/max bounds from the spec.
    Uses scipy.stats.truncnorm when bounds are present, and
    scipy.stats.norm otherwise.

    Args:
        spec: A validated NormalColumnSpec instance.
    """

    def __init__(self, spec: NormalColumnSpec):
        self.mean = spec.mean
        self.std = spec.std
        self.min = spec.min
        self.max = spec.max

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        if self.min is not None or self.max is not None:
            a = (self.min - self.mean) / self.std if self.min is not None else -np.inf
            b = (self.max - self.mean) / self.std if self.max is not None else np.inf
            return stats.truncnorm.rvs(a, b, loc=self.mean, scale=self.std, size=n, random_state=rng)
        return stats.norm.rvs(loc=self.mean, scale=self.std, size=n, random_state=rng)


class UniformGenerator(BaseGenerator):
    """Generator for uniformly distributed values.

    Args:
        spec: A validated UniformColumnSpec instance.
    """

    def __init__(self, spec: UniformColumnSpec):
        self.low = spec.low
        self.high = spec.high

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return stats.uniform.rvs(loc=self.low, scale=self.high - self.low, size=n, random_state=rng)


class LognormalGenerator(BaseGenerator):
    """Generator for lognormally distributed values.

    Args:
        spec: A validated LognormalColumnSpec instance.
    """

    def __init__(self, spec: LognormalColumnSpec):
        self.mu = spec.mu
        self.sigma = spec.sigma
        self.min = spec.min
        self.max = spec.max

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        values = stats.lognorm.rvs(s=self.sigma, scale=np.exp(self.mu), size=n, random_state=rng)
        if self.min is not None or self.max is not None:
            lower = self.min if self.min is not None else 0
            upper = self.max if self.max is not None else np.inf
            values = np.clip(values, lower, upper)
        return values


class BetaGenerator(BaseGenerator):
    """Generator for beta-distributed values.

    Args:
        spec: A validated BetaColumnSpec instance.
    """

    def __init__(self, spec: BetaColumnSpec):
        self.alpha = spec.alpha
        self.beta_param = spec.beta_param
        self.low = spec.low
        self.high = spec.high

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        values = stats.beta.rvs(self.alpha, self.beta_param, size=n, random_state=rng)
        return self.low + values * (self.high - self.low)


class GammaGenerator(BaseGenerator):
    """Generator for gamma-distributed values.

    Args:
        spec: A validated GammaColumnSpec instance.
    """

    def __init__(self, spec: GammaColumnSpec):
        self.shape = spec.shape
        self.scale = spec.scale

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return stats.gamma.rvs(a=self.shape, scale=self.scale, size=n, random_state=rng)


class ExponentialGenerator(BaseGenerator):
    """Generator for exponentially distributed values.

    Delegates to GammaGenerator with shape=1.

    Args:
        spec: A validated ExponentialColumnSpec instance.
    """

    def __init__(self, spec: ExponentialColumnSpec):
        gamma_spec = GammaColumnSpec(
            name=spec.name,
            type="gamma",
            shape=1,
            scale=spec.scale,
        )
        self._inner = GammaGenerator(gamma_spec)

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return self._inner.generate(n, rng, context)


class ParetoGenerator(BaseGenerator):
    """Generator for Pareto-distributed values.

    Args:
        spec: A validated ParetoColumnSpec instance.
    """

    def __init__(self, spec: ParetoColumnSpec):
        self.alpha = spec.alpha
        self.scale = spec.scale

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return stats.pareto.rvs(b=self.alpha, scale=self.scale, size=n, random_state=rng)


class StudentTGenerator(BaseGenerator):
    """Generator for Student's t-distributed values.

    Args:
        spec: A validated StudentTColumnSpec instance.
    """

    def __init__(self, spec: StudentTColumnSpec):
        self.df = spec.df
        self.loc = spec.loc
        self.scale = spec.scale

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return stats.t.rvs(df=self.df, loc=self.loc, scale=self.scale, size=n, random_state=rng)


class PoissonGenerator(BaseGenerator):
    """Generator for Poisson-distributed integer values.

    Args:
        spec: A validated PoissonColumnSpec instance.
    """

    def __init__(self, spec: PoissonColumnSpec):
        self.mu = spec.mu

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return stats.poisson.rvs(mu=self.mu, size=n, random_state=rng)


class BinomialGenerator(BaseGenerator):
    """Generator for binomially distributed integer values.

    Args:
        spec: A validated BinomialColumnSpec instance.
    """

    def __init__(self, spec: BinomialColumnSpec):
        self.n = spec.n
        self.p = spec.p

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return stats.binom.rvs(n=self.n, p=self.p, size=n, random_state=rng)


register("normal", NormalGenerator)
register("uniform", UniformGenerator)
register("lognormal", LognormalGenerator)
register("beta", BetaGenerator)
register("gamma", GammaGenerator)
register("exponential", ExponentialGenerator)
register("pareto", ParetoGenerator)
register("student_t", StudentTGenerator)
register("poisson", PoissonGenerator)
register("binomial", BinomialGenerator)
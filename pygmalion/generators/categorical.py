"""Generator for categorical columns."""

import numpy as np

from pygmalion.generators.base import BaseGenerator
from pygmalion.generators.registry import register
from pygmalion.schema.spec import CategoricalColumnSpec


class CategoricalGenerator(BaseGenerator):
    """Generator for categorical (string) values.

    Samples from a fixed set of values using the provided rng.
    If weights are provided, they are used as sampling probabilities.

    Args:
        spec: A validated CategoricalColumnSpec instance.
    """

    def __init__(self, spec: CategoricalColumnSpec):
        self.values = spec.values
        self.weights = spec.weights

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        return rng.choice(self.values, size=n, p=self.weights)


register("categorical", CategoricalGenerator)
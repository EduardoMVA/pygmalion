"""Generator for bootstrap-resampled columns."""

import numpy as np

from pygmalion.generators.base import BaseGenerator
from pygmalion.generators.registry import register
from pygmalion.schema.spec import BootstrapColumnSpec


class BootstrapGenerator(BaseGenerator):
    """Generator that resamples with replacement from observed values.

    Args:
        spec: A validated BootstrapColumnSpec instance.
    """

    def __init__(self, spec: BootstrapColumnSpec):
        self.values = np.array(spec.values)

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        indices = rng.integers(0, len(self.values), size=n)
        return self.values[indices]


register("bootstrap", BootstrapGenerator)
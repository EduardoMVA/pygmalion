"""Generator for mixture-of-distributions columns."""

import numpy as np

from pygmalion.generators.base import BaseGenerator
from pygmalion.generators.registry import register
from pygmalion.generators.numeric import NormalGenerator, UniformGenerator
from pygmalion.schema.spec import (
    MixtureColumnSpec,
    NormalColumnSpec,
    NormalComponentSpec,
    UniformColumnSpec,
    UniformComponentSpec,
)


class MixtureGenerator(BaseGenerator):
    """Generator that combines multiple distribution components.

    Allocates rows to components proportionally to their weights,
    generates each block, then concatenates and shuffles.

    Args:
        spec: A validated MixtureColumnSpec instance.
    """

    def __init__(self, spec: MixtureColumnSpec):
        self.weights = [c.weight for c in spec.components]
        self.generators = []

        for component in spec.components:
            if isinstance(component, NormalComponentSpec):
                col_spec = NormalColumnSpec(
                    name="_mixture_component",
                    type="normal",
                    mean=component.mean,
                    std=component.std,
                )
                self.generators.append(NormalGenerator(col_spec))
            elif isinstance(component, UniformComponentSpec):
                col_spec = UniformColumnSpec(
                    name="_mixture_component",
                    type="uniform",
                    low=component.low,
                    high=component.high,
                )
                self.generators.append(UniformGenerator(col_spec))

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        counts = rng.multinomial(n, self.weights)
        parts = []

        for gen, count in zip(self.generators, counts):
            if count > 0:
                parts.append(gen.generate(count, rng))

        result = np.concatenate(parts)
        rng.shuffle(result)
        return result


register("mixture", MixtureGenerator)
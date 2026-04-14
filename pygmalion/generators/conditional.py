"""Generator for conditional columns."""

import numpy as np
from scipy import stats

from pygmalion.generators.base import BaseGenerator
from pygmalion.generators.registry import register
from pygmalion.schema.spec import ConditionalColumnSpec


class ConditionalGenerator(BaseGenerator):
    """Generator whose distribution depends on another column's values.

    For each unique value in the condition column, applies a
    different distribution to the corresponding rows.

    Args:
        spec: A validated ConditionalColumnSpec instance.
    """

    def __init__(self, spec: ConditionalColumnSpec):
        self.condition_column = spec.condition_column
        self.cases = spec.cases

    def generate(self, n: int, rng: np.random.Generator, context=None) -> np.ndarray:
        if context is None or self.condition_column not in context:
            raise ValueError(
                f"ConditionalGenerator necesita la columna '{self.condition_column}' en el contexto"
            )

        condition_values = context[self.condition_column]
        result = np.empty(n, dtype=float)

        for case_value, case_spec in self.cases.items():
            mask = condition_values == case_value
            count = np.sum(mask)

            if count == 0:
                continue

            if case_spec.type == "normal":
                result[mask] = stats.norm.rvs(
                    loc=case_spec.mean, scale=case_spec.std, size=count, random_state=rng
                )
            elif case_spec.type == "uniform":
                result[mask] = stats.uniform.rvs(
                    loc=case_spec.low, scale=case_spec.high - case_spec.low, size=count, random_state=rng
                )
            elif case_spec.type == "categorical":
                result = result.astype(object)
                result[mask] = rng.choice(
                    case_spec.values, size=count, p=case_spec.weights
                )

        return result


register("conditional", ConditionalGenerator)
"""Base interface for all Pygmalion generators."""

from abc import ABC, abstractmethod
import numpy as np


class BaseGenerator(ABC):
    """Abstract base class for all data generators.

    Every generator must inherit from this class and implement
    the generate() method. This ensures a consistent interface
    that the engine can rely on.

    Subclasses receive a validated column spec in their constructor
    and produce a numpy array of values in generate().
    """

    @abstractmethod
    def generate(
        self,
        n: int,
        rng: np.random.Generator,
        context: dict[str, np.ndarray] = None,
    ) -> np.ndarray:
        """Generate n synthetic values.

        Args:
            n: Number of values to generate.
            rng: Numpy random generator instance for reproducibility.
            context: Optional dictionary of already-generated columns
                as numpy arrays. Used by conditional generators that
                need to reference other columns.

        Returns:
            A numpy array of length n with the generated values.
        """
        pass
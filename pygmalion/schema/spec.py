"""Pydantic models for validating Pygmalion JSON specs.

This module defines the schema that users must follow when
writing table specifications. Each column type has its own
model with specific validation rules.
"""

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Discriminator, Field, Tag, model_validator


class NormalColumnSpec(BaseModel):
    """Specification for a column with normal distribution.

    Generates numeric values following a Gaussian distribution.
    Supports optional truncation via min/max bounds.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "normal".
        mean: Mean of the distribution.
        std: Standard deviation. Must be positive.
        min: Optional lower bound for truncation.
        max: Optional upper bound for truncation.
            If both min and max are set, min must be less than max.

    Example:
        >>> col = NormalColumnSpec(
        ...     name="price", type="normal",
        ...     mean=100, std=15, min=0, max=200
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["normal"]
    mean: float
    std: float = Field(gt=0)
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_min_max(self):
        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError("min debe ser menor que max")
        return self

    
class LognormalColumnSpec(BaseModel):
    """Specification for a column with lognormal distribution.

    Generates positive values with right skew. Parameters mu and
    sigma refer to the mean and standard deviation of the
    underlying normal distribution in log-space.

    The actual generated values have median = exp(mu) and are
    always positive.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "lognormal".
        mu: Mean of the underlying normal in log-space.
        sigma: Standard deviation in log-space. Must be positive.
        min: Optional lower bound for truncation.
        max: Optional upper bound for truncation.
            If both are set, min must be less than max.

    Example:
        >>> col = LognormalColumnSpec(
        ...     name="salary", type="lognormal",
        ...     mu=10.5, sigma=0.5
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["lognormal"]
    mu: float
    sigma: float = Field(gt=0)
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_min_max(self):
        if self.min is not None and self.max is not None:
            if self.min >= self.max:
                raise ValueError("min debe ser menor que max")
        return self

class BetaColumnSpec(BaseModel):
    """Specification for a column with beta distribution.

    Generates values in a bounded range. By default [0, 1], but
    can be rescaled to any [low, high] interval.

    Useful for proportions, probabilities, rates, and any
    value naturally bounded between two limits.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "beta".
        alpha: First shape parameter. Must be positive.
        beta_param: Second shape parameter. Must be positive.
            Named beta_param to avoid conflict with the type name.
        low: Lower bound of the output range. Defaults to 0.
        high: Upper bound of the output range. Defaults to 1.
            Must be greater than low.

    Example:
        >>> col = BetaColumnSpec(
        ...     name="rate", type="beta",
        ...     alpha=2, beta_param=5
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["beta"]
    alpha: float = Field(gt=0)
    beta_param: float = Field(gt=0)
    low: float = 0
    high: float = 1

    @model_validator(mode="after")
    def validate_low_high(self):
        if self.low >= self.high:
            raise ValueError("low debe ser menor que high")
        return self


class GammaColumnSpec(BaseModel):
    """Specification for a column with gamma distribution.

    Generates positive values with right skew. Useful for
    waiting times, costs, amounts, and any positive continuous
    data.

    The exponential distribution is a special case with shape=1.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "gamma".
        shape: Shape parameter (k). Must be positive. Controls
            the form of the distribution.
        scale: Scale parameter (theta). Must be positive.
            The mean of the distribution is shape * scale.

    Example:
        >>> col = GammaColumnSpec(
        ...     name="wait_time", type="gamma",
        ...     shape=2, scale=10
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["gamma"]
    shape: float = Field(gt=0)
    scale: float = Field(gt=0)


class ExponentialColumnSpec(BaseModel):
    """Specification for a column with exponential distribution.

    Generates positive values modeling time between events.
    This is a special case of the gamma distribution with shape=1.

    The user can specify either scale (mean) or rate (1/mean),
    but not both. If rate is provided, scale is computed as 1/rate.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "exponential".
        scale: Mean of the distribution (1/rate). Must be positive.
        rate: Rate parameter (1/scale). Must be positive.
            Provide scale or rate, not both.

    Example:
        >>> col = ExponentialColumnSpec(
        ...     name="wait", type="exponential", scale=5.0
        ... )
        >>> # Equivalent using rate:
        >>> col = ExponentialColumnSpec(
        ...     name="wait", type="exponential", rate=0.2
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["exponential"]
    scale: float | None = Field(default=None, gt=0)
    rate: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_scale_or_rate(self):
        if self.scale is None and self.rate is None:
            raise ValueError("Debe proporcionar scale o rate")
        if self.scale is not None and self.rate is not None:
            raise ValueError("Proporcione scale o rate, no ambos")
        if self.rate is not None:
            self.scale = 1.0 / self.rate
        return self


class ParetoColumnSpec(BaseModel):
    """Specification for a column with Pareto distribution.

    Generates positive values with heavy right tail. Models
    phenomena following the 80/20 principle: wealth, city sizes,
    file sizes, website popularity.

    All generated values are >= scale.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "pareto".
        alpha: Shape parameter. Must be positive. Lower values
            produce heavier tails (more extreme outliers).
        scale: Minimum possible value. Must be positive.
            All generated values are >= scale.

    Example:
        >>> col = ParetoColumnSpec(
        ...     name="income", type="pareto",
        ...     alpha=2.5, scale=1000
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["pareto"]
    alpha: float = Field(gt=0)
    scale: float = Field(gt=0)


class StudentTColumnSpec(BaseModel):
    """Specification for a column with Student's t-distribution.

    Generates values similar to a normal distribution but with
    heavier tails. Useful for data with more frequent outliers
    than a normal would produce.

    As df increases, the distribution approaches a normal.
    With small df (3-5), tails are noticeably heavier.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "student_t".
        df: Degrees of freedom. Must be positive. Lower values
            produce heavier tails.
        loc: Center of the distribution. Defaults to 0.
        scale: Scale parameter. Must be positive. Defaults to 1.

    Example:
        >>> col = StudentTColumnSpec(
        ...     name="returns", type="student_t",
        ...     df=5, loc=0, scale=1
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["student_t"]
    df: float = Field(gt=0)
    loc: float = 0
    scale: float = Field(default=1, gt=0)


class PoissonColumnSpec(BaseModel):
    """Specification for a column with Poisson distribution.

    Generates non-negative integer values representing counts
    of events. Useful for number of calls per hour, errors
    per page, arrivals per minute.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "poisson".
        mu: Expected number of events (mean = variance).
            Must be positive.

    Example:
        >>> col = PoissonColumnSpec(
        ...     name="num_calls", type="poisson", mu=5
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["poisson"]
    mu: float = Field(gt=0)


class BinomialColumnSpec(BaseModel):
    """Specification for a column with binomial distribution.

    Generates integer values representing the number of
    successes in n independent trials, each with probability p.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "binomial".
        n: Number of trials. Must be a positive integer.
        p: Probability of success per trial.
            Must be between 0 (exclusive) and 1 (inclusive).

    Example:
        >>> col = BinomialColumnSpec(
        ...     name="num_approved", type="binomial",
        ...     n=20, p=0.8
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["binomial"]
    n: int = Field(gt=0)
    p: float = Field(gt=0, le=1)


class UniformColumnSpec(BaseModel):
    """Specification for a column with uniform distribution.

    Generates numeric values uniformly distributed between
    low and high bounds.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "uniform".
        low: Lower bound of the distribution.
        high: Upper bound. Must be greater than low.

    Example:
        >>> col = UniformColumnSpec(
        ...     name="age", type="uniform", low=18, high=65
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def validate_low_high(self):
        if self.low >= self.high:
            raise ValueError("low debe ser menor que high")
        return self

class CategoricalColumnSpec(BaseModel):
    """Specification for a categorical column.

    Generates string values sampled from a fixed set.
    Optionally accepts weights for non-uniform sampling.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "categorical".
        values: List of possible values. Must have at least one.
        weights: Optional list of probabilities. If provided,
            must have the same length as values, all non-negative,
            and sum to 1.0.

    Example:
        >>> col = CategoricalColumnSpec(
        ...     name="color", type="categorical",
        ...     values=["red", "blue"], weights=[0.7, 0.3]
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["categorical"]
    values: list[str] = Field(min_length=1)
    weights: list[float] | None = None

    @model_validator(mode="after")
    def validate_weights(self):
        if self.weights is not None:
            if len(self.weights) != len(self.values):
                raise ValueError("weights debe tener la misma longitud que values")
            if abs(sum(self.weights) - 1.0) > 1e-6:
                raise ValueError("weights debe sumar 1.0")
            if any(w < 0 for w in self.weights):
                raise ValueError("weights no puede tener valores negativos")
        return self


class NormalComponentSpec(BaseModel):
    """A normal distribution component within a mixture.

    Attributes:
        type: Must be "normal".
        mean: Mean of the component.
        std: Standard deviation. Must be positive.
        weight: Proportion of this component in the mixture.
            Must be between 0 (exclusive) and 1 (inclusive).
    """
    type: Literal["normal"]
    mean: float
    std: float = Field(gt=0)
    weight: float = Field(gt=0, le=1)


class UniformComponentSpec(BaseModel):
    """A uniform distribution component within a mixture.

    Attributes:
        type: Must be "uniform".
        low: Lower bound.
        high: Upper bound. Must be greater than low.
        weight: Proportion of this component in the mixture.
            Must be between 0 (exclusive) and 1 (inclusive).
    """
    type: Literal["uniform"]
    low: float
    high: float
    weight: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_low_high(self):
        if self.low >= self.high:
            raise ValueError("low debe ser menor que high")
        return self


MixtureComponentSpec = Annotated[
    Union[
        Annotated[NormalComponentSpec, Tag("normal")],
        Annotated[UniformComponentSpec, Tag("uniform")],
    ],
    Discriminator("type")
]


class MixtureColumnSpec(BaseModel):
    """Specification for a mixture-of-distributions column.

    Generates numeric values by sampling from multiple
    component distributions according to their weights.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "mixture".
        components: List of component specs. Must have at least 2.
            Each component has its own type, parameters, and weight.
            Weights must sum to 1.0.

    Example:
        >>> col = MixtureColumnSpec(
        ...     name="salary", type="mixture",
        ...     components=[
        ...         {"type": "normal", "mean": 30000, "std": 5000, "weight": 0.7},
        ...         {"type": "normal", "mean": 80000, "std": 15000, "weight": 0.3},
        ...     ]
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["mixture"]
    components: list[MixtureComponentSpec] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_weights(self):
        total = sum(c.weight for c in self.components)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("los weights de los componentes deben sumar 1.0")
        return self
    
class DerivedColumnSpec(BaseModel):
    """Specification for a column derived from other columns.

    The column value is computed by evaluating a Python expression
    that references other columns by name. Only arithmetic
    operations are supported.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "derived".
        expr: Python expression to evaluate. Column names are
            used as variables (e.g., "price * quantity").
        dependencies: Explicit list of column names that this
            column depends on. Must have at least one.

    Example:
        >>> col = DerivedColumnSpec(
        ...     name="total", type="derived",
        ...     expr="price * quantity",
        ...     dependencies=["price", "quantity"]
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["derived"]
    expr: str = Field(min_length=1)
    dependencies: list[str] = Field(min_length=1)


class ConditionalCaseSpec(BaseModel):
    """Distribution parameters for one case of a conditional column.

    A flexible model that holds parameters for normal, uniform,
    or categorical distributions. The model_validator ensures
    that the correct parameters are present for the declared type.

    Attributes:
        type: Distribution type ("normal", "uniform", or "categorical").
        mean: Mean (required for normal).
        std: Standard deviation (required for normal, must be positive).
        low: Lower bound (required for uniform).
        high: Upper bound (required for uniform).
        values: Possible values (required for categorical).
        weights: Optional probabilities (for categorical).
    """
    type: Literal["normal", "uniform", "categorical"]
    mean: float | None = None
    std: float | None = Field(default=None, gt=0)
    low: float | None = None
    high: float | None = None
    values: list[str] | None = None
    weights: list[float] | None = None

    @model_validator(mode="after")
    def validate_params(self):
        if self.type == "normal":
            if self.mean is None or self.std is None:
                raise ValueError("normal requiere mean y std")
        elif self.type == "uniform":
            if self.low is None or self.high is None:
                raise ValueError("uniform requiere low y high")
            if self.low >= self.high:
                raise ValueError("low debe ser menor que high")
        elif self.type == "categorical":
            if self.values is None or len(self.values) == 0:
                raise ValueError("categorical requiere values no vacío")
        return self


class ConditionalColumnSpec(BaseModel):
    """Specification for a column with conditional distributions.

    The distribution of values depends on the value of another
    column. Each possible value of the condition column maps
    to its own distribution.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "conditional".
        condition_column: Name of the column to condition on.
            Must exist in the same spec.
        cases: Dictionary mapping condition values to their
            distribution specs. Must have at least one case.

    Example:
        >>> col = ConditionalColumnSpec(
        ...     name="salary", type="conditional",
        ...     condition_column="level",
        ...     cases={
        ...         "junior": {"type": "normal", "mean": 25000, "std": 3000},
        ...         "senior": {"type": "normal", "mean": 60000, "std": 10000},
        ...     }
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["conditional"]
    condition_column: str = Field(min_length=1)
    cases: dict[str, ConditionalCaseSpec] = Field(min_length=1)


class BootstrapColumnSpec(BaseModel):
    """Specification for a bootstrap-resampled column.

    Generates values by sampling with replacement from a
    user-provided list of observed values. Preserves the
    empirical distribution without assuming any parametric form.

    Attributes:
        name: Column name. Must be non-empty.
        type: Must be "bootstrap".
        values: List of observed values to resample from.
            Must have at least one element. Can be numbers
            or strings.

    Example:
        >>> col = BootstrapColumnSpec(
        ...     name="price", type="bootstrap",
        ...     values=[100, 150, 200, 180, 120]
        ... )
    """
    name: str = Field(min_length=1)
    type: Literal["bootstrap"]
    values: list = Field(min_length=1)

ColumnSpec = Annotated[
    Union[
        Annotated[NormalColumnSpec, Tag("normal")],
        Annotated[UniformColumnSpec, Tag("uniform")],
        Annotated[CategoricalColumnSpec, Tag("categorical")],
        Annotated[MixtureColumnSpec, Tag("mixture")],
        Annotated[DerivedColumnSpec, Tag("derived")],
        Annotated[ConditionalColumnSpec, Tag("conditional")],
        Annotated[BootstrapColumnSpec, Tag("bootstrap")],
        Annotated[LognormalColumnSpec, Tag("lognormal")],
        Annotated[BetaColumnSpec, Tag("beta")],
        Annotated[GammaColumnSpec, Tag("gamma")],
        Annotated[ExponentialColumnSpec, Tag("exponential")],
        Annotated[ParetoColumnSpec, Tag("pareto")],
        Annotated[StudentTColumnSpec, Tag("student_t")],
        Annotated[PoissonColumnSpec, Tag("poisson")],
        Annotated[BinomialColumnSpec, Tag("binomial")],
    ],
    Discriminator("type")
]


class TableSpec(BaseModel):
    """Top-level specification for a synthetic table.

    Defines the number of rows, the list of columns with their
    distributions, and optional constraints that the generated
    data must satisfy.

    Attributes:
        num_rows: Number of rows to generate. Must be positive.
        columns: List of column specifications. Must have at least
            one. Each column can be any supported distribution type.
        constraints: Optional list of boolean expressions that
            all rows must satisfy (e.g., "age >= 18").
        seed: Optional random seed for reproducibility. If provided,
            the same seed always produces the same output.

    Example:
        >>> spec = TableSpec(
        ...     num_rows=1000,
        ...     columns=[
        ...         {"name": "x", "type": "normal", "mean": 0, "std": 1}
        ...     ],
        ...     seed=42
        ... )
    """
    num_rows: int = Field(gt=0)
    columns: list[ColumnSpec] = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    seed: int | None = None
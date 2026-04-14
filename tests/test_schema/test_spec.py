import pytest
from pydantic import ValidationError
from pygmalion.schema.spec import NormalColumnSpec, TableSpec


def test_spec_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            NormalColumnSpec(name="precio", type="normal", mean=25000, std=5000)
        ]
    )
    assert spec.num_rows == 100
    assert spec.columns[0].name == "precio"


def test_std_negativo():
    with pytest.raises(ValidationError):
        NormalColumnSpec(name="precio", type="normal", mean=25000, std=-5)


def test_std_cero():
    with pytest.raises(ValidationError):
        NormalColumnSpec(name="precio", type="normal", mean=25000, std=0)


def test_num_rows_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=-10,
            columns=[
                NormalColumnSpec(name="precio", type="normal", mean=25000, std=5000)
            ]
        )


def test_columns_vacio():
    with pytest.raises(ValidationError):
        TableSpec(num_rows=100, columns=[])


def test_falta_mean():
    with pytest.raises(ValidationError):
        NormalColumnSpec(name="precio", type="normal", std=5000)


def test_name_vacio():
    with pytest.raises(ValidationError):
        NormalColumnSpec(name="", type="normal", mean=100, std=10)

def test_uniform_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "edad", "type": "uniform", "low": 18, "high": 65}
        ]
    )
    assert spec.columns[0].type == "uniform"


def test_uniform_low_mayor_que_high():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "edad", "type": "uniform", "low": 65, "high": 18}
            ]
        )


def test_categorical_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "color", "type": "categorical", "values": ["rojo", "azul", "verde"]}
        ]
    )
    assert spec.columns[0].type == "categorical"


def test_categorical_con_weights():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {
                "name": "color",
                "type": "categorical",
                "values": ["rojo", "azul"],
                "weights": [0.7, 0.3]
            }
        ]
    )
    assert spec.columns[0].weights == [0.7, 0.3]


def test_categorical_weights_no_suman_uno():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {
                    "name": "color",
                    "type": "categorical",
                    "values": ["rojo", "azul"],
                    "weights": [0.5, 0.3]
                }
            ]
        )


def test_categorical_weights_longitud_diferente():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {
                    "name": "color",
                    "type": "categorical",
                    "values": ["rojo", "azul"],
                    "weights": [0.5, 0.3, 0.2]
                }
            ]
        )


def test_normal_con_truncamiento():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "precio", "type": "normal", "mean": 50, "std": 10, "min": 0, "max": 100}
        ]
    )
    assert spec.columns[0].min == 0
    assert spec.columns[0].max == 100


def test_truncamiento_min_mayor_que_max():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "precio", "type": "normal", "mean": 50, "std": 10, "min": 100, "max": 0}
            ]
        )


def test_spec_columnas_mixtas():
    spec = TableSpec(
        num_rows=500,
        columns=[
            {"name": "precio", "type": "normal", "mean": 100, "std": 15},
            {"name": "edad", "type": "uniform", "low": 18, "high": 65},
            {"name": "color", "type": "categorical", "values": ["rojo", "azul"]},
        ]
    )
    assert len(spec.columns) == 3


def test_type_invalido():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "inventado", "mean": 0, "std": 1}
            ]
        )

def test_mixture_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {
                "name": "salario",
                "type": "mixture",
                "components": [
                    {"type": "normal", "mean": 30000, "std": 5000, "weight": 0.7},
                    {"type": "normal", "mean": 80000, "std": 15000, "weight": 0.3},
                ]
            }
        ]
    )
    assert len(spec.columns[0].components) == 2


def test_mixture_weights_no_suman_uno():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {
                    "name": "salario",
                    "type": "mixture",
                    "components": [
                        {"type": "normal", "mean": 30000, "std": 5000, "weight": 0.5},
                        {"type": "normal", "mean": 80000, "std": 15000, "weight": 0.3},
                    ]
                }
            ]
        )


def test_mixture_un_solo_componente():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {
                    "name": "salario",
                    "type": "mixture",
                    "components": [
                        {"type": "normal", "mean": 30000, "std": 5000, "weight": 1.0},
                    ]
                }
            ]
        )


def test_mixture_componentes_mixtos():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {
                "name": "valor",
                "type": "mixture",
                "components": [
                    {"type": "normal", "mean": 50, "std": 10, "weight": 0.6},
                    {"type": "uniform", "low": 0, "high": 100, "weight": 0.4},
                ]
            }
        ]
    )
    assert spec.columns[0].components[1].type == "uniform"


def test_derived_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "precio", "type": "normal", "mean": 100, "std": 10},
            {"name": "cantidad", "type": "uniform", "low": 1, "high": 10},
            {
                "name": "total",
                "type": "derived",
                "expr": "precio * cantidad",
                "dependencies": ["precio", "cantidad"],
            },
        ]
    )
    assert spec.columns[2].type == "derived"
    assert spec.columns[2].dependencies == ["precio", "cantidad"]


def test_derived_sin_dependencias():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {
                    "name": "total",
                    "type": "derived",
                    "expr": "precio * 2",
                    "dependencies": [],
                }
            ]
        )


def test_derived_expr_vacio():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {
                    "name": "total",
                    "type": "derived",
                    "expr": "",
                    "dependencies": ["precio"],
                }
            ]
        )


def test_conditional_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "nivel", "type": "categorical", "values": ["junior", "senior"]},
            {
                "name": "salario",
                "type": "conditional",
                "condition_column": "nivel",
                "cases": {
                    "junior": {"type": "normal", "mean": 25000, "std": 3000},
                    "senior": {"type": "normal", "mean": 60000, "std": 10000},
                },
            },
        ],
    )
    assert spec.columns[1].type == "conditional"


def test_conditional_cases_vacio():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {
                    "name": "salario",
                    "type": "conditional",
                    "condition_column": "nivel",
                    "cases": {},
                }
            ],
        )

def test_bootstrap_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "x", "type": "bootstrap", "values": [1, 2, 3, 4, 5]}
        ],
    )
    assert spec.columns[0].type == "bootstrap"


def test_bootstrap_values_vacio():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "bootstrap", "values": []}
            ],
        )

def test_lognormal_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "salario", "type": "lognormal", "mu": 10.5, "sigma": 0.5}
        ],
    )
    assert spec.columns[0].type == "lognormal"


def test_lognormal_sigma_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "lognormal", "mu": 0, "sigma": -1}
            ],
        )

def test_beta_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "tasa", "type": "beta", "alpha": 2, "beta_param": 5}
        ],
    )
    assert spec.columns[0].type == "beta"


def test_beta_alpha_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "beta", "alpha": -1, "beta_param": 5}
            ],
        )


def test_beta_low_mayor_que_high():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "beta", "alpha": 2, "beta_param": 5, "low": 10, "high": 5}
            ],
        )

def test_gamma_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "tiempo", "type": "gamma", "shape": 2, "scale": 10}
        ],
    )
    assert spec.columns[0].type == "gamma"


def test_gamma_shape_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "gamma", "shape": -1, "scale": 10}
            ],
        )


def test_gamma_scale_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "gamma", "shape": 2, "scale": -5}
            ],
        )

def test_exponential_con_scale():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "t", "type": "exponential", "scale": 5.0}
        ],
    )
    assert spec.columns[0].scale == 5.0


def test_exponential_con_rate():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "t", "type": "exponential", "rate": 0.2}
        ],
    )
    assert abs(spec.columns[0].scale - 5.0) < 1e-6


def test_exponential_sin_scale_ni_rate():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "t", "type": "exponential"}
            ],
        )


def test_exponential_con_ambos():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "t", "type": "exponential", "scale": 5, "rate": 0.2}
            ],
        )

def test_pareto_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "ingreso", "type": "pareto", "alpha": 2.5, "scale": 1000}
        ],
    )
    assert spec.columns[0].type == "pareto"


def test_pareto_alpha_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "pareto", "alpha": -1, "scale": 100}
            ],
        )


def test_pareto_scale_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "pareto", "alpha": 2, "scale": -100}
            ],
        )

def test_student_t_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "retorno", "type": "student_t", "df": 5, "loc": 0, "scale": 1}
        ],
    )
    assert spec.columns[0].type == "student_t"


def test_student_t_defaults():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "retorno", "type": "student_t", "df": 5}
        ],
    )
    assert spec.columns[0].loc == 0
    assert spec.columns[0].scale == 1


def test_student_t_df_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "student_t", "df": -1}
            ],
        )

def test_poisson_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "llamadas", "type": "poisson", "mu": 5}
        ],
    )
    assert spec.columns[0].type == "poisson"


def test_poisson_mu_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "poisson", "mu": -1}
            ],
        )


def test_binomial_valido():
    spec = TableSpec(
        num_rows=100,
        columns=[
            {"name": "aprobados", "type": "binomial", "n": 20, "p": 0.8}
        ],
    )
    assert spec.columns[0].type == "binomial"


def test_binomial_p_fuera_de_rango():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "binomial", "n": 10, "p": 1.5}
            ],
        )


def test_binomial_n_negativo():
    with pytest.raises(ValidationError):
        TableSpec(
            num_rows=100,
            columns=[
                {"name": "x", "type": "binomial", "n": -5, "p": 0.5}
            ],
        )
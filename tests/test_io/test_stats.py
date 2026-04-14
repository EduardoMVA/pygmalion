from pygmalion.io.stats import stats_only


def test_stats_normal():
    spec = {
        "num_rows": 100,
        "columns": [
            {"name": "precio", "type": "normal", "mean": 100, "std": 10}
        ],
    }
    result = stats_only(spec)
    assert result["num_rows"] == 100
    assert result["num_columns"] == 1
    assert result["columns"]["precio"]["type"] == "normal"
    assert result["columns"]["precio"]["mean"] == 100


def test_stats_uniform():
    spec = {
        "num_rows": 50,
        "columns": [
            {"name": "edad", "type": "uniform", "low": 18, "high": 65}
        ],
    }
    result = stats_only(spec)
    assert result["columns"]["edad"]["expected_mean"] == 41.5


def test_stats_categorical():
    spec = {
        "num_rows": 50,
        "columns": [
            {"name": "color", "type": "categorical", "values": ["r", "g", "b"]}
        ],
    }
    result = stats_only(spec)
    assert result["columns"]["color"]["num_categories"] == 3
    assert result["columns"]["color"]["distribution"] == "uniform"


def test_stats_mixture():
    spec = {
        "num_rows": 100,
        "columns": [
            {
                "name": "salario",
                "type": "mixture",
                "components": [
                    {"type": "normal", "mean": 30000, "std": 5000, "weight": 0.5},
                    {"type": "normal", "mean": 70000, "std": 10000, "weight": 0.5},
                ],
            }
        ],
    }
    result = stats_only(spec)
    assert result["columns"]["salario"]["expected_mean"] == 50000.0


def test_stats_derived():
    spec = {
        "num_rows": 100,
        "columns": [
            {"name": "a", "type": "uniform", "low": 0, "high": 1},
            {
                "name": "b",
                "type": "derived",
                "expr": "a * 2",
                "dependencies": ["a"],
            },
        ],
    }
    result = stats_only(spec)
    assert result["columns"]["b"]["expr"] == "a * 2"


def test_stats_conditional():
    spec = {
        "num_rows": 100,
        "columns": [
            {"name": "nivel", "type": "categorical", "values": ["jr", "sr"]},
            {
                "name": "salario",
                "type": "conditional",
                "condition_column": "nivel",
                "cases": {
                    "jr": {"type": "normal", "mean": 25000, "std": 3000},
                    "sr": {"type": "normal", "mean": 60000, "std": 10000},
                },
            },
        ],
    }
    result = stats_only(spec)
    assert result["columns"]["salario"]["condition_column"] == "nivel"
    assert "jr" in result["columns"]["salario"]["cases"]


def test_stats_spec_completo():
    spec = {
        "num_rows": 500,
        "columns": [
            {"name": "a", "type": "normal", "mean": 0, "std": 1},
            {"name": "b", "type": "uniform", "low": 0, "high": 100},
            {"name": "c", "type": "categorical", "values": ["x", "y"]},
        ],
    }
    result = stats_only(spec)
    assert result["num_columns"] == 3
    assert len(result["columns"]) == 3
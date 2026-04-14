def test_import_pygmalion():
    import pygmalion
    assert pygmalion is not None

def test_import_subpackages():
    from pygmalion import schema, generators, engine, io, constraints
    assert all(m is not None for m in [schema, generators, engine, io, constraints])

def test_api_publica():
    from pygmalion import (
        synthesize,
        learn_from_csv,
        template_from_data,
        stats_only,
        quality_report,
        to_csv,
        to_json,
        TableSpec,
    )
    assert all(
        callable(f)
        for f in [synthesize, learn_from_csv, template_from_data,
                  stats_only, quality_report, to_csv, to_json]
    )
    assert TableSpec is not None


def test_version():
    import pygmalion
    assert hasattr(pygmalion, "__version__")
    assert pygmalion.__version__ == "0.1.0"
"""Pygmalion: Synthetic tabular data generation from JSON specs."""

from pygmalion.engine.synthesizer import synthesize
from pygmalion.io.reader import learn_from_csv, template_from_data
from pygmalion.io.stats import stats_only
from pygmalion.io.quality import quality_report
from pygmalion.io.writer import to_csv, to_json
from pygmalion.schema.spec import TableSpec

__version__ = "0.1.0"

__all__ = [
    "synthesize",
    "learn_from_csv",
    "template_from_data",
    "stats_only",
    "quality_report",
    "to_csv",
    "to_json",
    "TableSpec",
]
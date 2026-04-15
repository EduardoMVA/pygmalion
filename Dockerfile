FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY pygmalion/ pygmalion/

RUN pip install --no-cache-dir ".[dev]"

COPY tests/ tests/
COPY examples/ examples/

CMD ["pytest", "tests/", "-v"]
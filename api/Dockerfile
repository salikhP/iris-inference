FROM python:3.12-slim

RUN pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --no-interaction --no-ansi

COPY . /app

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

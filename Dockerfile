FROM python:3.9 AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

FROM base as poetry
ENV POETRY_VERSION=1.2.2

# System deps:
COPY poetry.lock pyproject.toml ./
COPY zlw ./zlw

RUN python -m pip install "poetry==$POETRY_VERSION" && \
    python -m poetry export --without dev --without-hashes --no-interaction --no-ansi -f requirements.txt -o requirements.txt

FROM base as runtime
COPY --from=poetry /app /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*
RUN python -m pip install -r requirements.txt && \
    python -m pip install tzdata
COPY . .

RUN mkdir data
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
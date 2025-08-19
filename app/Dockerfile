FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=2.1.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Update system packages and install useful packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl ca-certificates\
    && rm -rf /var/lib/apt/lists/*

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Create non-root user
RUN useradd -m appuser

WORKDIR /app

# Copy poetry files first
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . /app

# Expose Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Drop privileges
USER appuser

# Run Streamlit directly since packages are installed globally
CMD ["poetry", "run", "streamlit", "run", "entrypoint.py", "--server.port=8501", "--server.address=0.0.0.0"]

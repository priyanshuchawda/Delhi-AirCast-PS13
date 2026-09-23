FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src/delhi_aircast ./src/delhi_aircast
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "delhi_aircast.api:app", "--host", "0.0.0.0", "--port", "8000"]

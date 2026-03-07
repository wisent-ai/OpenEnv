FROM openenv-base AS builder

WORKDIR /app/env

COPY pyproject.toml .
ARG BUILD_MODE=production

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY . .

FROM python:3.11-slim AS runtime

WORKDIR /app/env

COPY --from=builder /app/env /app/env

ENV PATH="/app/env/.venv/bin:$PATH" \
    PYTHONPATH=/app/env

EXPOSE 8000

HEALTHCHECK --interval=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "env.app:app", "--host", "0.0.0.0", "--port", "8000"]

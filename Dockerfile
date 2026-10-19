FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Install dependencies first so source changes do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project

COPY alembic.ini ./
# 迁移脚本位于 src/oce/alembic，随 src 一并拷贝
COPY src ./src

# Hatchling force-include sources required to build the local project.
COPY bench/datasets ./bench/datasets
COPY bench/profiles/local.toml ./bench/profiles/local.toml
COPY bench/profiles/docker.example.toml ./bench/profiles/docker.example.toml
COPY bench/profiles/sweep_topk.example.toml ./bench/profiles/sweep_topk.example.toml

RUN uv sync --locked --no-dev

EXPOSE 8986

CMD ["uv", "run", "uvicorn", "oce.main:app", "--host", "0.0.0.0", "--port", "8986"]

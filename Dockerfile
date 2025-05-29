# --------------------------------------------------------------------------- #
# forecast_vol; CPU-only container image
# --------------------------------------------------------------------------- #
# Build:  docker build -t forecast_vol .
# Run :  docker run --rm -v "$PWD/data:/app/data" forecast_vol make pipeline
# --------------------------------------------------------------------------- #
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- 1. OS build deps ----------------------------------------------------- #
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. copy metadata *only*  (better cache hits) ------------------------ #
COPY pyproject.toml ./

# ---- 3. install runtime deps --------------------------------------------- #
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --config-settings editable=false .

# ---- 4. now add the source tree & configs -------------------------------- #
COPY src ./src
COPY configs ./configs
COPY Makefile .

# ---- 5. sane default cmd -------------------------------------------------- #
# (Drop into an interactive shell by default; users can still run
#  `docker ... forecast_vol make pipeline` verbatim.)
CMD ["python", "-m", "forecast_vol.market.build_sessions", "--help"]

# Runtime versions used by the benchmark agent.
FROM python:3.12.13-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
        cmdstanpy==1.3.0 \
        matplotlib==3.11.1 \
        numpy==2.5.1 \
        pandas==3.0.5 \
        scipy==1.18.0 \
    && python -m cmdstanpy.install_cmdstan --version 2.39.0 --cores 2

WORKDIR /workspace

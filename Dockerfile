FROM python:3.8-slim-buster

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=TRUE

# Install the virtual environment
ARG PIPENV_DEV=false
WORKDIR /app

COPY Pipfile Pipfile.lock ./
# TODO: Stop ignoring when hadolint/hadolint#511 is resolved
# hadolint ignore=DL3013,DL3042
RUN pip install --no-cache-dir pipenv==2021.5.29 && \
    pipenv install --system --deploy && \
    mkdir -p "/root/.cache/black/19.10b0"  # Temporary workaround for psf/black#1143

# Copy config and code
COPY blindbot blindbot/

ENTRYPOINT [ "python", "blindbot" ]

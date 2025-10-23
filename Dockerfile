# Dockerfile - Predictive Customer Churn Project
# Maintainer: Nishant Kamal (nishant2081)
FROM python:3.12-slim
LABEL maintainer="nishant2081"
# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
# Python Dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt
COPY . /app
# Ensure writable plots directory
RUN mkdir -p /app/src/plots && chmod -R 777 /app/src/plots
ENV PREFECT_API_URL="" \
    PREFECT_API_ENABLE="false" \
    PREFECT_TEST_MODE="true" \
    PREFECT_LOCAL_MODE="true" \
    PREFECT_LOGGING_LEVEL="INFO" \
    PREFECT_ANALYTICS_ENABLED="false" \
    DATA_PATH="/app/data/customer_churn.csv"
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
# Create Non-Root User (safe runtime)
RUN useradd -m appuser && chown -R appuser /app
USER appuser
CMD ["/app/entrypoint.sh"]

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/pyvis

RUN pip install --no-cache-dir pandas matplotlib

COPY runner.py /opt/pyvis/runner.py

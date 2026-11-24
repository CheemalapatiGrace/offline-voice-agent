FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .
RUN useradd --create-home --uid 10001 agent && mkdir -p /data && chown agent /data
USER agent
ENV VA_DB_PATH=/data/voice_agent.db VA_HOST=0.0.0.0 VA_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"
CMD ["python", "-m", "voice_agent.server"]

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements.hybrid.txt ./
ARG SUPPORT_HYBRID=false
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$SUPPORT_HYBRID" = "true" ]; then pip install --no-cache-dir -r requirements.hybrid.txt; fi
COPY app ./app
COPY data ./data
RUN useradd --create-home --uid 10001 support && mkdir /data && chown support:support /data
USER support
ENV SUPPORT_DB_PATH=/data/support.db SUPPORT_CONFIG_PATH=/run/secrets/support-users.json
EXPOSE 8090
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8090"]

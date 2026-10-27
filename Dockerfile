FROM python:3.13-slim

RUN pip install --no-cache-dir pandas==3.1.0 pyarrow==26.0.0

WORKDIR /app

USER 10001:10001

CMD ["python", "--version"]

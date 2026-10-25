
FROM python:3.13-slim

RUN pip install --no-cache-dir pandas pyarrow

WORKDIR /app

USER 10001:10001

CMD ["python", "--version"]

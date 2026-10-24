# The course pins MLflow 3.16.1; there is no official image for it, so build one.
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.16.1 psycopg2-binary==2.9.13 boto3==1.43.118
EXPOSE 5000

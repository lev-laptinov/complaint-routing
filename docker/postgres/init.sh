#!/bin/sh
# One PostgreSQL 16 instance, one database per consumer.
set -e
for db in airflow mlflow warehouse; do
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -c "CREATE DATABASE $db"
done

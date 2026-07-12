#!/bin/bash -ex

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "localhost,127.0.0.1,localhost:*,127.0.0.1:*,host.docker.internal:*"

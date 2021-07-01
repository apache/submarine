#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# description: Start and stop daemon script for.
#

set -euo pipefail

# Check if the bucket "minio/mlflow" already exists
check_minio_mlflow_bucket_exists() {
    if /bin/bash -c "./mc ls minio/mlflow" >/dev/null 2>&1; then
       true
    else
       false
    fi
}

MLFLOW_S3_ENDPOINT_URL="http://submarine-minio-service:9000"
AWS_ACCESS_KEY_ID="submarine_minio"
AWS_SECRET_ACCESS_KEY="submarine_minio"
BACKEND_URI="mysql+pymysql://mlflow:password@localhost:3306/mlflow"
DEFAULT_ARTIFACT_ROOT="s3://mlflow"
STATIC_PREFIX="/mlflow"

/bin/bash -c "sqlite3 store.db"

/bin/bash -c "sleep 60; ./mc config host add minio ${MLFLOW_S3_ENDPOINT_URL} ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY}"

if ! check_minio_mlflow_bucket_exists; then
	/bin/bash -c "./mc mb minio/mlflow"
else
	echo "Bucket minio/mlflow already exists, skipping creation."
fi

/bin/bash -c "mlflow server --host 0.0.0.0 --backend-store-uri ${BACKEND_URI} --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} --static-prefix ${STATIC_PREFIX}"
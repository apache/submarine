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

set -euo pipefail

S3_ENDPOINT_URL="http://submarine-minio-service:9000"
AWS_ACCESS_KEY_ID="submarine_minio"
AWS_SECRET_ACCESS_KEY="submarine_minio"


# Wait for minio pod to setup
/bin/bash -c "kubectl wait --for=condition=ready pod -l app=submarine-minio-pod; mc config host add minio ${S3_ENDPOINT_URL} ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY}"

# Create if the bucket "minio/submarine" not exists

if /bin/bash -c "mc ls minio/submarine" >/dev/null 2>&1; then
    echo "Bucket minio/submarine already exists, skipping creation."
else
    /bin/bash -c "mc mb minio/submarine"
fi

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

BACKEND_URI="mysql+pymysql://mlflow:password@submarine-database:3306/mlflowdb"
DEFAULT_ARTIFACT_ROOT="s3://mlflow"
STATIC_PREFIX="/mlflow"

/bin/bash -c "mlflow server --host 0.0.0.0 --backend-store-uri ${BACKEND_URI} --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} --static-prefix ${STATIC_PREFIX}"
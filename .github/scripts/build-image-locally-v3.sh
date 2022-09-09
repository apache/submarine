#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

SUBMARINE_VERSION="0.8.0-SNAPSHOT"
FOLDER_LIST=("database" "mlflow" "submarine" "operator-v3" "agent" "experiment-prehandler")
IMAGE_LIST=(
  "apache/submarine:database-${SUBMARINE_VERSION}"
  "apache/submarine:mlflow-${SUBMARINE_VERSION}"
  "apache/submarine:server-${SUBMARINE_VERSION}"
  "apache/submarine:operator-${SUBMARINE_VERSION}"
  "apache/submarine:agent-${SUBMARINE_VERSION}"
  "apache/submarine:experiment-prehandler-${SUBMARINE_VERSION}"
)

for i in "${!IMAGE_LIST[@]}"
do
  echo "Build Image ${IMAGE_LIST[i]}"
  echo "Execute ./dev-support/docker-images/${FOLDER_LIST[i]}/build.sh"
  ./dev-support/docker-images/"${FOLDER_LIST[i]}"/build.sh
  kind load docker-image "${IMAGE_LIST[i]}"
done


#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euxo pipefail

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"

BASE_FOLDER=../../submarine-server/server-api/src/main/java
PACKAGE_PATH=org/apache/submarine/server/api/proto

# Please also match the setting in the proto files.
PROTO_DIR_NAME="proto"
protoc --java_out=${BASE_FOLDER} --proto_path=${BASE_FOLDER} ${BASE_FOLDER}/${PACKAGE_PATH}/model_config.proto

echo "Insert apache license at the top of file ..."
for filename in $(find ${BASE_FOLDER}/${PACKAGE_PATH}/*.java -type f); do
  
  echo "$filename"
  cat license-header.txt "$filename" > "${filename}_tmp"
  rm "$filename"
  mv "${filename}_tmp" "${filename}"
done

set +euxo pipefail
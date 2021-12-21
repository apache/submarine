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

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"

SUBMARINE_PROJECT_PATH="$FWDIR/../.."
SWAGGER_JAR_URL="https://repo1.maven.org/maven2/org/openapitools/openapi-generator-cli/4.3.1/openapi-generator-cli-4.3.1.jar"
SWAGGER_CODEGEN_JAR="openapi-generator-cli.jar"
SWAGGER_CODEGEN_CONF="swagger_config.json"
SWAGGER_CODEGEN_FILE="openapi.json"
SDK_OUTPUT_PATH="sdk/python"

# Please also match with the content in `swagger_config.json`
API_DIR_NAME="client"

echo "API_DIR_NAME ${API_DIR_NAME}"
# clear SDK_OUTPUT_PATH
rm -rf "${SDK_OUTPUT_PATH}/*"

submarine_dist_exists=$(find -L "${SUBMARINE_PROJECT_PATH}/submarine-dist/target" -name "submarine-dist-*.tar.gz")
# Build source code if the package doesn't exist.
if [[ -z "${submarine_dist_exists}" ]]; then
  cd "${SUBMARINE_PROJECT_PATH}"
  mvn clean package -DskipTests
  cd "$FWDIR" # go back to FWDIR after packaging
fi

echo "Generating openAPI 3.0 definition file ..."
# TODO(pingsutw): generate openapi.json without starting submarine server
bash ${SUBMARINE_PROJECT_PATH}/submarine-dist/target//submarine-dist-*/submarine-dist-*/bin/submarine-daemon.sh start getMysqlJar
sleep 5
rm openapi.json
wget http://localhost:8080/v1/openapi.json
bash ${SUBMARINE_PROJECT_PATH}/submarine-dist/target//submarine-dist-*/submarine-dist-*/bin/submarine-daemon.sh stop

openapi_generator_cli_exists=$(find -L "${FWDIR}" -name "openapi-generator-cli*")
if [[ -z "${openapi_generator_cli_exists}" ]]; then
  echo "Downloading the swagger-codegen JAR package ..."
  wget -O "${SWAGGER_CODEGEN_JAR}" "${SWAGGER_JAR_URL}"
fi

echo "Generating Python SDK for Submarine ..."
rm -r sdk/
java -jar ${SWAGGER_CODEGEN_JAR} generate \
     -i "${SWAGGER_CODEGEN_FILE}" \
     -g python \
     -o ${SDK_OUTPUT_PATH} \
     -c ${SWAGGER_CODEGEN_CONF}

echo "Insert apache license at the top of file ..."
for filename in $(find ${SDK_OUTPUT_PATH}/submarine/${API_DIR_NAME} -type f); do
  echo "$filename"
  cat license-header.txt "$filename" > "${filename}_1"
  rm "$filename"
  mv "${filename}_1" "${filename}"
done

echo "Move APIs to pysubmarine"
cp -r ${SDK_OUTPUT_PATH}/submarine/${API_DIR_NAME} ${SUBMARINE_PROJECT_PATH}/submarine-sdk/pysubmarine/submarine/

echo "Fix Python SDK code style"
${SUBMARINE_PROJECT_PATH}/dev-support/style-check/python/auto-format.sh

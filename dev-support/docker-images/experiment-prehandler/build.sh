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

SUBMARINE_VERSION=0.8.0-SNAPSHOT
SUBMARINE_IMAGE_NAME="apache/submarine:experiment-prehandler-${SUBMARINE_VERSION}"

export CURRENT_PATH=$(cd "${PWD}">/dev/null; pwd)
export SUBMARINE_HOME=${CURRENT_PATH}/../../..

mkdir -p "${CURRENT_PATH}/tmp"

cp -r ${SUBMARINE_HOME}/submarine-experiment-prehandler ${CURRENT_PATH}/tmp/
cp ${SUBMARINE_HOME}/bin/experiment-prehandler.sh ${CURRENT_PATH}/tmp/submarine-experiment-prehandler/

HADOOP_TAR_URL="https://dlcdn.apache.org/hadoop/common/hadoop-3.3.3/hadoop-3.3.3.tar.gz"
tmpfile=$(mktemp)
trap "test -f $tmpfile && rm $tmpfile" RETURN
curl -L -o $tmpfile ${HADOOP_TAR_URL}
mv $tmpfile ${CURRENT_PATH}/tmp/hadoop-3.3.3.tar.gz

curl -L -o ${CURRENT_PATH}/tmp/hadoop-aws-3.3.3.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.3/hadoop-aws-3.3.3.jar 
curl -L -o ${CURRENT_PATH}/tmp/aws-java-sdk-bundle-1.12.267.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.267/aws-java-sdk-bundle-1.12.267.jar

echo "Start building the ${SUBMARINE_IMAGE_NAME} docker image ..."
docker build -t ${SUBMARINE_IMAGE_NAME} .

# clean temp file
rm -rf "${CURRENT_PATH}/tmp"

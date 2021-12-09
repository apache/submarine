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

SUBMARINE_VERSION=0.7.0-SNAPSHOT
SUBMARINE_IMAGE_NAME="apache/submarine:sidecar-agent-${SUBMARINE_VERSION}"

if [ -L ${BASH_SOURCE-$0} ]; then
  PWD=$(dirname $(readlink "${BASH_SOURCE-$0}"))
else
  PWD=$(dirname ${BASH_SOURCE-$0})
fi
export CURRENT_PATH=$(cd "${PWD}">/dev/null; pwd)
export SUBMARINE_HOME=${CURRENT_PATH}/../../..

if [ ! -d "${SUBMARINE_HOME}/submarine-server/server-submitter/submarine-k8s-agent/target" ]; then
  mkdir -p "${SUBMARINE_HOME}/submarine-server/server-submitter/submarine-k8s-agent/target"
fi
submarine_dist_exists=$(find -L "${SUBMARINE_HOME}/submarine-server/server-submitter/submarine-k8s-agent/target" -name "submarine-k8s-agent-${SUBMARINE_VERSION}.tar.gz")
# Build source code if the package doesn't exist.
if [[ -z "${submarine_dist_exists}" ]]; then
  cd "${SUBMARINE_HOME}/submarine-server/server-submitter/submarine-k8s-agent"
  mvn clean package -DskipTests
fi

mkdir -p "${CURRENT_PATH}/tmp"
cp ${SUBMARINE_HOME}/submarine-server/server-submitter/submarine-k8s-agent/target/submarine-k8s-agent-${SUBMARINE_VERSION}.tar.gz "${CURRENT_PATH}/tmp"

# download mysql connect java
MYSQL_VERSION=5.1.39
MYSQL_JAR_URL="https://repo1.maven.org/maven2/mysql/mysql-connector-java/${MYSQL_VERSION}/mysql-connector-java-${MYSQL_VERSION}.jar"
tmpfile=$(mktemp)
trap "test -f $tmpfile && rm $tmpfile" RETURN
curl -L -o $tmpfile ${MYSQL_JAR_URL}
mv $tmpfile ${CURRENT_PATH}/tmp/mysql-connector-java-${MYSQL_VERSION}.jar

# Replace the mysql jdbc.url in the submarine-site.xml file with the link name of the submarine container
# `submarine-database` is submarine database container name
cp ${SUBMARINE_HOME}/conf/submarine-site.xml "${CURRENT_PATH}/tmp/"
sed -i.bak 's/127.0.0.1:3306/submarine-database:3306/g' "${CURRENT_PATH}/tmp/submarine-site.xml"

# build image
cd ${CURRENT_PATH}
echo "Start building the ${SUBMARINE_IMAGE_NAME} docker image ..."
docker build -t ${SUBMARINE_IMAGE_NAME} .

# clean temp file
rm -rf "${CURRENT_PATH}/tmp"

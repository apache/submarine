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

# Please make sure you've already built whole package once before executing this script!

set -e


if [ -L ${BASH_SOURCE-$0} ]; then
  PWD=$(dirname $(readlink "${BASH_SOURCE-$0}"))
else
  PWD=$(dirname ${BASH_SOURCE-$0})
fi

export CURRENT_PATH=$(cd "${PWD}">/dev/null; pwd)
export SUBMARINE_HOME=${CURRENT_PATH}/../..

NAMESPACE=$1

[[ ! -f mvnw ]] && mvn -N io.takari:maven:0.7.7:wrapper -Dmaven=3.6.1

if [ ! -d "${SUBMARINE_HOME}/submarine-dist/target" ]; then
  mkdir "${SUBMARINE_HOME}/submarine-dist/target"
fi

# Build submarine-server module
cd ${SUBMARINE_HOME}/submarine-server
${SUBMARINE_HOME}/submarine-cloud-v2/hack/mvnw clean package -DskipTests

# Build assemble tar ball
cd ${SUBMARINE_HOME}/submarine-dist
${SUBMARINE_HOME}/submarine-cloud-v2/hack/mvnw clean package -DskipTests

eval $(minikube docker-env)

${SUBMARINE_HOME}/dev-support/docker-images/submarine/build.sh
# Delete the deployment and the operator will create a new one using new image
kubectl delete -n "$NAMESPACE" deployments submarine-server
eval $(minikube docker-env -u)

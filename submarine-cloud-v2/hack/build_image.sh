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

set -e
NAMESPACE="submarine-user-test"

# ========================================

help_message() {
  cat <<< "\
Usage: $0 [image]

image:
    all             build all images
    server          build submarine server
    database        build submarine database
    jupyter         build submarine jupyter-notebook
    jupyter-gpu     build submarine jupyter-notebook-gpu
    mlflow          build submarine mlflow"
}

build_server() {
  [[ ! -f mvnw ]] && mvn -N io.takari:maven:0.7.7:wrapper -Dmaven=3.6.1
  eval $(minikube docker-env -u)
  ./mvnw clean package -DskipTests
  eval $(minikube docker-env)
  ./dev-support/docker-images/submarine/build.sh
  # Delete the deployment and the operator will create a new one using new image
  kubectl delete -n "$NAMESPACE" deployments submarine-server
  eval $(minikube docker-env -u)
}

build_database() {
  eval $(minikube docker-env)
  ./dev-support/docker-images/database/build.sh
  # Delete the deployment and the operator will create a new one using new image
  kubectl delete -n "$NAMESPACE" deployments submarine-database
  eval $(minikube docker-env -u)
}

build_jupyter() {
  eval $(minikube docker-env)
  ./dev-support/docker-images/jupyter/build.sh
  eval $(minikube docker-env -u)
}

build_jupyter_gpu() {
  eval $(minikube docker-env)
  ./dev-support/docker-images/jupyter-gpu/build.sh
  eval $(minikube docker-env -u)
}

build_mlflow() {
  eval $(minikube docker-env)
  ./dev-support/docker-images/mlflow/build.sh
  # Delete the deployment and the operator will create a new one using new image
  kubectl delete -n "$NAMESPACE" deployments submarine-mlflow
  eval $(minikube docker-env -u)
}

# ========================================

if [[ "$#" -ne 1 ]]; then
  help_message >&2
  exit 1
fi

SUBMARINE_HOME=`git rev-parse --show-toplevel`
cd "$SUBMARINE_HOME"

case "$1" in
  "all")
    build_server
    build_database
    build_jupyter
    build_jupyter_gpu
    build_mlflow
    ;;
  "server")
    build_server
    ;;
  "database")
    build_database
    ;;
  "jupyter")
    build_jupyter
    ;;
  "jupyter-gpu")
    build_jupyter_gpu
    ;;
  "mlflow")
    build_mlflow
    ;;
  *)
    help_message >&2
    exit 1
    ;;
esac

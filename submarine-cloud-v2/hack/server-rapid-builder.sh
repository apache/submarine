#!/usr/bin/env bash
set -e

export CURRENT_PATH=$(cd "${PWD}">/dev/null; pwd)
export SUBMARINE_HOME=${CURRENT_PATH}/../../..
NAMESPACE=$1


[[ ! -f mvnw ]] && mvn -N io.takari:maven:0.7.7:wrapper -Dmaven=3.6.1
eval $(minikube docker-env -u)

export CURRENT_PATH=$(cd "${PWD}">/dev/null; pwd)
export SUBMARINE_HOME=${CURRENT_PATH}/../..

if [ ! -d "${SUBMARINE_HOME}/submarine-dist/target" ]; then
  mkdir "${SUBMARINE_HOME}/submarine-dist/target"
fi

cd ${SUBMARINE_HOME}/submarine-server
mvn clean package -DskipTests
cd ${SUBMARINE_HOME}/submarine-dist
mvn clean package -DskipTests

[[ ! -f mvnw ]] && mvn -N io.takari:maven:0.7.7:wrapper -Dmaven=3.6.1
eval $(minikube docker-env -u)
eval $(minikube docker-env)
${SUBMARINE_HOME}/dev-support/docker-images/submarine/build.sh
# Delete the deployment and the operator will create a new one using new image
kubectl delete -n "$NAMESPACE" deployments submarine-server
eval $(minikube docker-env -u)

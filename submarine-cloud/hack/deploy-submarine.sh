#!/bin/bash
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
set -e

ROOT=$(unset CDPATH && cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
cd $ROOT
SUBMARINE_HOME=${ROOT}/..

source $ROOT/hack/lib.sh

hack::check_requirements

# install submarine in k8s cluster
function install_submarine() {
  if [ ! -d "${ROOT}/hack/conf" ]; then
    mkdir "${ROOT}/hack/conf"
  fi

  if [ ! -f "${ROOT}/hack/conf/submarine-site.xml" ]; then
    cp "${SUBMARINE_HOME}/conf/submarine-site.xml.template" "${ROOT}/hack/conf/submarine-site.xml"

    # Replace the mysql jdbc.url in the submarine-site.xml file with the name of the submarine database ip/service
    sed -i.bak "s/127.0.0.1:3306/${DATABASE_IP}:3306/g" "${ROOT}/hack/conf/submarine-site.xml"
  fi

  if [ ! -f "${ROOT}/hack/conf/log4j.properties" ]; then
    cp ${SUBMARINE_HOME}/conf/log4j.properties.template ${ROOT}/hack/conf/log4j.properties
  fi

  echo ""
  echo -e "Have you configured the \033[31m${ROOT}/hack/conf/submarine-site.xml\033[0m file?"
  echo -e "Have you configured the \033[31m${ROOT}/hack/conf/log4j.properties\033[0m file?"
  echo -n "Do you want to deploy submarine in k8s cluster now? [y/n]"
  read myselect
  if [[ "$myselect" == "y" || "$myselect" == "Y" ]]; then
    if kubectl get configmap --namespace default | grep submarine-config >/dev/null ; then
      kubectl delete configmap --namespace default submarine-config
    fi
    kubectl create configmap --namespace default submarine-config --from-file=${ROOT}/hack/conf/submarine-site.xml --from-file=${ROOT}/hack/conf/log4j.properties

    docker pull apache/submarine:operator-0.3.0-SNAPSHOT
    kind load docker-image apache/submarine:operator-0.3.0-SNAPSHOT
    kubectl apply -f $ROOT/manifests/submarine-operator/

    docker pull apache/submarine:database-0.3.0-SNAPSHOT
    kind load docker-image apache/submarine:database-0.3.0-SNAPSHOT
    docker pull apache/submarine:server-0.3.0-SNAPSHOT
    kind load docker-image apache/submarine:server-0.3.0-SNAPSHOT
    kubectl apply -f $ROOT/manifests/submarine-cluster/

    cat <<EOF
NOTE: You can open your browser and access the submarine workbench at http://127.0.0.1/
EOF
  fi
}

function uninstall_submarine() {
  if kubectl get configmap --namespace default | grep submarine-config >/dev/null ; then
    kubectl delete configmap --namespace default submarine-config
  fi
  kubectl delete -f $ROOT/manifests/submarine-operator/
  kubectl delete -f $ROOT/manifests/submarine-cluster/

  cat <<EOF
NOTE: Submarine cluster has been deleted
EOF
}

usage() {
    cat <<EOF
This script use kind to create Submarine cluster, about kind please refer: https://kind.sigs.k8s.io/
* This script will automatically install kubectr-${KUBECTL_VERSION} and kind-${KIND_VERSION} in ${OUTPUT_BIN}

Options:
       -d,--database           ip/service of submarine database, default value: submarine-database
       -u,--uninstall          uninstall submarine cluster
       -h,--help               prints the usage message
Usage:
    install: $0 --database database_ip
     OR
    unstall: $0 -u
EOF
}

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -d|--database)
    DATABASE_IP="$2"
    shift
    shift
    ;;
    -u|--uninstall)
    UNINSTALL="TRUE"
    shift
    ;;
    *)
    echo "unknown option: $key"
    usage
    exit 1
    ;;
esac
done

DATABASE_IP=${DATABASE_IP:-submarine-database}
echo "Submarine database ip: ${DATABASE_IP}"

if [[ "$UNINSTALL" == "TRUE" ]]; then
  uninstall_submarine
else
  install_submarine
fi

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
SUBMARINE_VERSION="0.4.0-SNAPSHOT"
TF_JUPYTER_IMAGE="apache/submarine:tf2.1.0-jupyter";

source $ROOT/hack/lib.sh

hack::ensure_kubectl

# Install submarine in k8s cluster
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

  if [[ "$TEST" == "" ]]; then
    echo ""
    echo -e "Have you configured the \033[31m${ROOT}/hack/conf/submarine-site.xml\033[0m file?"
    echo -e "Have you configured the \033[31m${ROOT}/hack/conf/log4j.properties\033[0m file?"
    echo -n "Do you want to deploy submarine in k8s cluster now? [y/n]"
    read myselect
  else
    myselect="y"
  fi

  if [[ "$myselect" == "y" || "$myselect" == "Y" ]]; then
    if $KUBECTL_BIN get configmap --namespace default | grep submarine-config >/dev/null ; then
      $KUBECTL_BIN delete configmap --namespace default submarine-config
    fi
    $KUBECTL_BIN create configmap --namespace default submarine-config --from-file=${ROOT}/hack/conf/submarine-site.xml --from-file=${ROOT}/hack/conf/log4j.properties

    if ! docker inspect apache/submarine:operator-${SUBMARINE_VERSION} >/dev/null ; then
      docker pull apache/submarine:operator-${SUBMARINE_VERSION}
    fi
    $KIND_BIN load docker-image apache/submarine:operator-${SUBMARINE_VERSION}
    $KUBECTL_BIN apply -f $ROOT/manifests/submarine-operator/

    if ! docker inspect apache/submarine:database-${SUBMARINE_VERSION} >/dev/null ; then
      docker pull apache/submarine:database-${SUBMARINE_VERSION}
    fi
    $KIND_BIN load docker-image apache/submarine:database-${SUBMARINE_VERSION}

    if ! docker inspect ${TF_JUPYTER_IMAGE} >/dev/null ; then
      docker pull ${TF_JUPYTER_IMAGE}
    fi
    $KIND_BIN load docker-image ${TF_JUPYTER_IMAGE} >/dev/null

    if ! docker inspect apache/submarine:server-${SUBMARINE_VERSION} >/dev/null ; then
      docker pull apache/submarine:server-${SUBMARINE_VERSION}
    fi
    $KIND_BIN load docker-image apache/submarine:server-${SUBMARINE_VERSION}
    $KUBECTL_BIN apply -f $ROOT/manifests/submarine-cluster/

    echo "NOTE: You can open your browser and access the submarine workbench at http://127.0.0.1/"
  fi
}

# Uninstall submarine in k8s cluster
function uninstall_submarine() {
  if $KUBECTL_BIN get configmap --namespace default | grep submarine-config >/dev/null ; then
    $KUBECTL_BIN delete configmap --namespace default submarine-config
  fi
  $KUBECTL_BIN delete -f $ROOT/manifests/submarine-operator/
  $KUBECTL_BIN delete -f $ROOT/manifests/submarine-cluster/

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
       -t,--test               auto install
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
    -t|--test)
    TEST="TRUE"
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

export KUBECONFIG=~/.kube/kind-config-${clusterName:-kind}

if [[ "$UNINSTALL" == "TRUE" ]]; then
  uninstall_submarine
else
  install_submarine
fi

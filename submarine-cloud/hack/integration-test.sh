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
set -eo pipefail
set -x

ROOT=$(unset CDPATH && cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
cd $ROOT
SUBMARINE_HOME=${ROOT}/..

source $ROOT/hack/lib.sh

hack::ensure_kubectl

export KUBECONFIG=~/.kube/kind-config-${clusterName:-kind}

function start() {
  $ROOT/hack/kind-cluster-build.sh
  $ROOT/hack/deploy-submarine.sh --test

  for((i=1;i<=30;i++)); do
    info=`curl -s -m 10 --connect-timeout 10 -I http://127.0.0.1/api/v1/cluster/address`
    code=`echo $info | grep "HTTP" | awk '{print $2}'`

    #############  DON'T DELETE NEXT DEBUG COMAND  #############
    # $KUBECTL_BIN get node
    # $KUBECTL_BIN get pods
    # $KUBECTL_BIN get svc
    # podname=`$KUBECTL_BIN get pods | grep submarinecluster-submarine | awk '{print $1}'`
    # $KUBECTL_BIN describe pod $podname
    # $KUBECTL_BIN logs $podname
    ############################################################

    if [ "$code" == "200" ];then
        echo "Start submarine on k8s success!"
        exit;
    else
        echo "Request failed with response code = $code"
    fi
    sleep 3
  done

  #############  DON'T DELETE NEXT DEBUG COMAND  #############
  # $KUBECTL_BIN get node
  # podname=`$KUBECTL_BIN get pods | grep submarinecluster-submarine | awk '{print $1}'`
  # $KUBECTL_BIN describe pod $podname
  # $KUBECTL_BIN exec -it $podname cat /opt/submarine-current/logs/submarine.log
  # $KUBECTL_BIN exec $podname -- bash -c "tail -500 /opt/submarine-current/logs/submarine.log"
  # $KUBECTL_BIN get pods | grep submarinecluster-submarine | awk '{print $1}' | xargs -I {} $KUBECTL_BIN exec {} -- bash -c "tail -500 /opt/submarine-current/logs/submarine.log"
  # kubectl get pods -n operations | grep operations | awk '{print $1}' | xargs -I {} kubectl exec -it -n operations {} cat /tmp/operations-server.INFO
  ############################################################
  echo "Stop submarine on k8s failure!"
}

function stop() {
  $ROOT/hack/kind delete cluster
}

function update_docker_images() {
  $SUBMARINE_HOME/dev-support/docker-images/database/build.sh
  $SUBMARINE_HOME/dev-support/docker-images/operator/build.sh
  $SUBMARINE_HOME/dev-support/docker-images/submarine/build.sh

  docker images
}

usage() {
    cat <<EOF
This script use kind to create Kubernetes cluster and deploy submarine to k8s

Options:
       -h,--help          prints the usage message
       -s,--start         Create k8s and start submarine
       -t,--stop          Delete k8s cluster and submarine
       -u,--update        update submarine docker image
Usage:
    $0 --start --update
EOF
}

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -s|--start)
    OPERATION="START"
    shift
    ;;
    -t|--stop)
    OPERATION="STOP"
    shift
    ;;
    -u|--update)
    UPDATE_IAMGE="TRUE"
    shift
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *)
    echo "unknown option: $key"
    usage
    exit 1
    ;;
esac
done

OPERATION=${OPERATION:-""}
UPDATE_IAMGE=${UPDATE_IAMGE:-""}

if [[ $UPDATE_IAMGE == "TRUE" ]]; then
  update_docker_images
fi

if [[ $OPERATION == "STOP" ]]; then
  stop
else
  start
fi

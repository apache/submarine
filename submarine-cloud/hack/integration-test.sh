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

function start() {
  $ROOT/hack/kind-cluster-build.sh
  $ROOT/hack/deploy-submarine.sh --test

  for((i=1;i<=100;i++)); do
    if curl http://127.0.0.1/api/v1/cluster/address | grep \"status\":\"OK\" ; then
      echo "Cluster start success!"
      exit;
    fi
    sleep 3
  done

  echo "Cluster start failure!"
}

function stop() {
  $ROOT/hack/kind delete cluster
}

if [[ "$1" == "stop" ]]; then
  stop
else
  start
fi

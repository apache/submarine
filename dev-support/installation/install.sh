#!/bin/bash
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

helm install submarine ./helm-charts/submarine

if [[ ! -f ./dev-support/installation/istioctl ]]; then
    wget https://github.com/istio/istio/releases/download/1.6.8/istio-1.6.8-linux-amd64.tar.gz
    tar zxvf istio-1.6.8-linux-amd64.tar.gz
    rm -rf istio-1.6.8-linux-amd64.tar.gz
    cp ./istio-1.6.8/bin/istioctl ./dev-support/installation/istioctl
    rm -rf istio-1.6.8
fi

./dev-support/installation/istioctl install --skip-confirmation

kubectl create ns seldon-system
kubectl apply -f ./dev-support/installation/seldon-secret.yaml
helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --namespace seldon-system \
    --version 1.10.0 \
    --set istio.enabled=true \
    --set executor.defaultEnvSecretRefName=seldon-core-init-container-secret 2> /dev/null
kubectl apply -f ./dev-support/installation/seldon-gateway.yaml

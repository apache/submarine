#!/usr/bin/env bash
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

wait_interval=5
wait_timeout=2000
submarine_user_namespace=default

wait_times=$((wait_timeout / wait_interval))

# Fix submarine-database start failed in kind. https://github.com/kubernetes/minikube/issues/7906
sudo ln -s /etc/apparmor.d/usr.sbin.mysqld /etc/apparmor.d/disable/
sudo apparmor_parser -R /etc/apparmor.d/usr.sbin.mysqld

kubectl create namespace submarine
kubectl create namespace "$submarine_user_namespace"
kubectl label namespace submarine istio-injection=enabled
kubectl label namespace "$submarine_user_namespace" istio-injection=enabled
kubectl apply -n submarine-user-test -f ./submarine-cloud-v2/artifacts/submarine/submarine-server.yaml  # For github action
kubectl apply
helm install --wait --set storageClass.provisioner=rancher.io/local-path --set storageClass.volumeBindingMode=WaitForFirstConsumer submarine ./helm-charts/submarine -n submarine
kubectl apply -f ./submarine-cloud-v2/artifacts/examples/example-submarine.yaml -n "$submarine_user_namespace"

# Polling waiting for the submarine to be in the RUNNING state
for ((i=0;i<$wait_times;++i)); do
  state=$(kubectl get submarine -n "$submarine_user_namespace" -o=jsonpath='{.items[0].status.submarineState.state}')
  if [[ "$state" == "RUNNING" ]]; then
    echo "Submarine is running!"
    kubectl describe submarine -n "$submarine_user_namespace"
    kubectl get all -n "$submarine_user_namespace"
    kubectl port-forward -n "$submarine_user_namespace" svc/submarine-database 3306:3306 &
    kubectl port-forward -n "$submarine_user_namespace" svc/submarine-server 8080:8080 &
    kubectl port-forward -n "$submarine_user_namespace" svc/submarine-minio-service 9000:9000 &
    kubectl port-forward -n "$submarine_user_namespace" svc/submarine-mlflow-service 5001:5000 &
    exit 0
  elif [[ "$state" == "FAILED" ]]; then
    echo "Submarine failed!" 1>&2
    kubectl describe submarine -n "$submarine_user_namespace"
    kubectl get all -n "$submarine_user_namespace"
    exit 1
  else
    sleep $wait_interval
  fi
done
echo "Timeout limit reached!" 1>&2
kubectl describe submarine -n "$submarine_user_namespace"
kubectl get all -n "$submarine_user_namespace"
exit 1

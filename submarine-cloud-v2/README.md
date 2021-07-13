<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements.  See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# submarine-cloud-v2

`submarine-cloud-v2` implements the operator for Submarine application.

## Why submarine-cloud-v2?

Because `submarine-cloud` is outdated, `submarine-cloud-v2` is the refactored version of `submarine-cloud`. In addition, after `submarine-cloud-v2` finishes, we will replace `submarine-cloud` with `submarine-cloud-v2`.

## Initialization

```bash
# Install dependencies
go mod vendor
# Run the cluster
minikube start --vm-driver=docker  --kubernetes-version v1.15.11
```

## Run operator out-of-cluster

```bash
# Step1: Build & Run "submarine-operator"
make
./submarine-operator

# Step2: Deploy a Submarine
kubectl apply -f artifacts/examples/crd.yaml
kubectl create ns submarine-user-test
kubectl apply -n submarine-user-test -f artifacts/examples/example-submarine.yaml

# Step3: Exposing Service
# Method1 -- use minikube ip
minikube ip  # you'll get the IP address of minikube, ex: 192.168.49.2

# Method2 -- use port-forwarding
kubectl port-forward --address 0.0.0.0 -n submarine-user-test service/traefik 32080:80

# Step4: View Workbench
# http://{minikube ip}:32080 (from Method 1), ex: http://192.168.49.2:32080
# or http://127.0.0.1:32080 (from Method 2).

# Step5: Delete Submarine
# By deleting the submarine custom resource, the operator will do the following things:
#   (1) Remove all relevant Helm chart releases
#   (2) Remove all resources in the namespace "submariner-user-test"
#   (3) Remove all non-namespaced resources (Ex: PersistentVolume) created by client-go API
#   (4) **Note:** The namespace "submarine-user-test" will not be deleted
kubectl delete submarine example-submarine -n submarine-user-test

# Step6: Stop the operator
# Press ctrl+c to stop the operator
```

## Run operator in-cluster

```bash
# Step1: Build image "submarine-operator" to minikube's Docker
eval $(minikube docker-env)
make image

# Step2: Setup RBAC (ClusterRole, ClusterRoleBinding, and ServiceAccount)
kubectl apply -f artifacts/examples/submarine-operator-service-account.yaml

# Step3: Deploy a submarine-operator
kubectl apply -f artifacts/examples/submarine-operator.yaml

# Step4: Deploy a submarine
kubectl apply -f artifacts/examples/crd.yaml
kubectl create ns submarine-user-test
kubectl apply -n submarine-user-test -f artifacts/examples/example-submarine.yaml

# Step5: Inspect the logs of submarine-operator
kubectl logs -f $(kubectl get pods --output=name | grep submarine-operator)

# Step6: Exposing Service
# Method1 -- use minikube ip
minikube ip  # you'll get the IP address of minikube, ex: 192.168.49.2

# Method2 -- use port-forwarding
kubectl port-forward --address 0.0.0.0 -n submarine-user-test service/traefik 32080:80

# Step7: View Workbench
# http://{minikube ip}:32080 (from Method 1), ex: http://192.168.49.2:32080
# or http://127.0.0.1:32080 (from Method 2).

# Step8: Delete Submarine
# By deleting the submarine custom resource, the operator will do the following things:
#   (1) Remove all relevant Helm chart releases
#   (2) Remove all resources in the namespace "submariner-user-test"
#   (3) Remove all non-namespaced resources (Ex: PersistentVolume) created by client-go API
#   (4) **Note:** The namespace "submarine-user-test" will not be deleted
kubectl delete submarine example-submarine -n submarine-user-test

# Step9: Delete the submarine-operator
kubectl delete deployment submarine-operator-demo
```

## Development

Please check out the [Developer Guide](./docs/developer-guide.md).

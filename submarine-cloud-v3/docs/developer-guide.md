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

# Developer Guide

Golang version: `1.17`

## Run operator out-of-cluster

```bash
# Step1: Run the operator in a terminal.
make install run

# Step2: Deploy a submarine.
kubectl create ns submarine-user-test
kubectl apply -n submarine-user-test -f config/samples/_v1alpha1_submarine.yaml

# Step3: Cleanup submarine.
kubectl delete -n submarine-user-test submarine example-submarine
kubectl delete ns submarine-user-test

# Step4: Cleanup operator.
# Just close the running process at Step1.
```

## Run operator in-cluster
```bash
# Step1: Build the docker image.
eval $(minikube docker-env)
make docker-build
eval $(minikube docker-env -u)

# Step2: Deploy the operator.
# A new namespace is created with name submarine-cloud-v3-system, and will be used for the deployment.
make deploy

# Step3: Verify the operator is up and running.
kubectl get deployment -n submarine-cloud-v3-system

# Step4: Deploy a submarine.
kubectl create ns submarine-user-test
kubectl apply -n submarine-user-test -f config/samples/_v1alpha1_submarine.yaml

# Step5: Cleanup submarine.
kubectl delete -n submarine-user-test submarine example-submarine
kubectl delete ns submarine-user-test

# Step6: Cleanup operator.
make undeploy
```

### Rebuild Operator Image

When running operator in-cluster , we need to rebuild the operator image for changes to take effect.

```bash
eval $(minikube docker-env)
make docker-build
eval $(minikube docker-env -u)
```

## Coding Style

For `go` files, please use [gofmt](https://golang.org/pkg/cmd/gofmt/) to format the code.

For `yaml` files, please use [prettier](https://prettier.io/) to format the code.

## Generate API

Steps to modify custom resource definition (CRD):
1. Modify the Submarine type in `api/v1alpha1/submarine_types.go`.
2. Run `make generate` to update `api/v1alpha1/zz_generated.deepcopy.go`.
3. Run `make manifests` to update crd file `config/crd/bases/submarine.apache.org_submarines.yaml`.
4. Modify the sample submarine `config/samples/_v1alpha1_submarine`.

One can add [marker comments](https://book.kubebuilder.io/reference/markers.html) in Go code to control the manifests generation.



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
Kubernetes version: `1.21.0`

## Prerequisites

First finish the prerequisites specified in the [QuickStart](https://submarine.apache.org/docs/next/gettingStarted/quickstart) section on the submarine website. Prepare a minikube cluster with Istio installed. Prepare namespaces `submarine-user-test` and `submarine-cloud-v3-system` and label them `istio-injection=enabled`.

Verify with kubectl:

```bash
$ kubectl get namespace --show-labels
NAME                        STATUS   AGE    LABELS
istio-system                Active   7d8h   kubernetes.io/metadata.name=istio-system
submarine-user-test         Active   27h    istio-injection=enabled,kubernetes.io/metadata.name=submarine-user-test
submarine-cloud-v3-system   Active   27h    istio-injection=enabled,kubernetes.io/metadata.name=submarine-submarine-cloud-v3-system

$ kubectl get pod -n istio-system
NAME                                    READY   STATUS    RESTARTS   AGE
istio-ingressgateway-77968dbd74-wq4vb   1/1     Running   1          7d4h
istiod-699b647f8b-nx9rt                 1/1     Running   2          7d4h
```

## Run operator out-of-cluster

Before running submarine operator, install submarine dependencies with helm. `--set dev=true` option will not install the operator deployment to the cluster.

```bash
helm install --set dev=true submarine ../helm-charts/submarine/ -n submarine-cloud-v3-system
```

Now we run the submarine operator.

```bash
# Step1: Apply the submarine CRD.
# Note that the submarine CRD /helm-charts/submarine/crds/crd.yaml is for submarine-cloud-v2.
# This step will overwrite it with a CRD generated with controller-gen.
make install

# Step2: Run the operator in a terminal.
make run

# Step3: Deploy a submarine.
kubectl apply -n submarine-user-test -f config/samples/_v1alpha1_submarine.yaml
```

If you follow the above steps, you can view the submarine workbench via the same approach specified in the [QuickStart](https://submarine.apache.org/docs/next/gettingStarted/quickstart) section on the submarine website.


```bash
# Step4: Cleanup submarine.
kubectl delete -n submarine-user-test submarine example-submarine

# Step5: Cleanup operator.
# Just close the running process at Step2.

# Step6: Delete the submarine CRD.
make uninstall
```

## Run operator in-cluster

Note that running `make deploy` `make undeploy` creates and deletes `submarine-cloud-v3-system` respectively. Therefore we will run `helm uninstall` before `make undeploy`.

```bash
# Step1: Build the docker image.
eval $(minikube docker-env)
make docker-build
eval $(minikube docker-env -u)

# Step2: Install submarine dependencies with Helm
# If the namespace submarine-cloud-v3-system doesn't exist yet,
# running Step3 will create it.
# However, note that if podSecurityPolicy is enabled,
# the submarine operator pod will not be permitted until running this
helm install --set dev=true submarine ../helm-charts/submarine/ -n submarine-cloud-v3-system

# Step3: Deploy the operator.
# 1) Note that the submarine CRD /helm-charts/submarine/crds/crd.yaml is for submarine-cloud-v2.
#    This step will overwrite it with a CRD generated with controller-gen.
# 2) A new namespace is created with name submarine-cloud-v3-system, and will be used for the deployment.
#    If such a namespace already exists, this step will overwrite its spec.
# 3) Other resources are created in this namespaces
make deploy

# Step4: Verify the operator is up and running.
kubectl get deployment -n submarine-cloud-v3-system

# Step5: Deploy a submarine.
kubectl apply -n submarine-user-test -f config/samples/_v1alpha1_submarine.yaml

# You can now view the submarine workbench

# Step6: Cleanup submarine.
kubectl delete -n submarine-user-test submarine example-submarine

# Step7: Cleanup submarine dependencies.
helm uninstall submarine -n submarine-cloud-v3-system

# Step8: Cleanup operator.
# Note that this step deletes the namespace submarine-cloud-v3-system
make undeploy
```

### Installing Helm in a different namespace

By default, the Istio virtual service created by the operator binds to the Istio gateway `submarine-cloud-v3-system/submarine-gateway`, which should be installed in the `submarine-cloud-v3-system` namespace via Helm. If `helm install` is run with `-n <your_namespace>`, edit the spec `spec.gateways` of the virtual service to be `<your_namespace>/submarine-gateway` manually once it's created by the operator.

### Rebuild Operator Image

When running operator in-cluster, we need to rebuild the operator image for changes to take effect.

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

## Add resource

Steps to add new resource created and controlled by the operator:
1. Add or modify yaml files under `artifacts/`.
2. Modify `createSubmarine` in `contorllers/submarine_controller.go` to create resource from yaml files.
3. If needed, modify reconcile logic in `contorllers/submarine_controller.go`.
4. If needed, import new scheme in `main.go`.
5. If there are new resource types, add RBAC marker comments in `contorllers/submarine_controller.go`.
6. Run `make manifests` to update cluster role rules in `config/rbac/role.yaml`.

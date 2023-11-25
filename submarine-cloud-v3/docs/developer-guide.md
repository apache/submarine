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

Golang version: `1.19`
Kubernetes version: `1.24` - `1.27`

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
helm dependency update ./helm-charts/submarine
helm install --set dev=true submarine ../helm-charts/submarine/ -n submarine-cloud-v3-system
```

Run the submarine operator.

```bash
# Step1: Apply the submarine CRD.
# Note that the submarine CRD /helm-charts/submarine/crds/crd.yaml is for submarine-cloud-v2.
# This step will overwrite it with a CRD generated with controller-gen.
make install

# Step2: Run the operator in a terminal.
make run

# Step3: Deploy a submarine.
kubectl apply -n submarine-user-test -f config/samples/_v1_submarine.yaml
```

Ensure that submarine is ready.

```bash
$ kubectl get pods -n submarine-cloud-v3-system
NAME                                                     READY   STATUS    RESTARTS   AGE
notebook-controller-deployment-5b489cf59d-52lff          1/1     Running   2          22h
submarine-cloud-v3-controller-manager-7b5787d8bc-hvpbk   2/2     Running   5          22h
training-operator-6dcd5b9c64-v7k7k                       1/1     Running   1          22h

$ kubectl get pods -n submarine-user-test
NAME                                     READY   STATUS    RESTARTS   AGE
submarine-database-0                     2/2     Running   0          24m
submarine-minio-6d757cc97c-qwft5         2/2     Running   0          24m
submarine-mlflow-657f5f8f6-9zxt6         2/2     Running   0          24m
submarine-server-6c787c69b5-pv2dr        2/2     Running   0          24m
submarine-tensorboard-7b447d94dd-d6kkm   2/2     Running   0          24m

$ kubectl get virtualservices -n submarine-user-test
NAME                        GATEWAYS                                          HOSTS                               AGE
submarine-virtual-service   ["submarine-cloud-v3-system/submarine-gateway"]   ["submarine-user-test.submarine"]   24m
```

The Istio virtual service now accepts traffic sent to destination `<submarine namespace>.submarine`. In our case it's `submarine-user-test.submarine`. Expose the service by forwarding traffic from local port 80 to the Istio ingress gateway.

```bash
# You may need to grant kubectl the capacity to bind to previleged ports without root.
# sudo setcap CAP_NET_BIND_SERVICE=+ep <full path to kubectl binary>

# Step4: Expose the service using kubectl port-forward
kubectl port-forward --address 127.0.0.1 -n istio-system service/istio-ingressgateway 80:80

# Alternatively, use minikube tunnel, which asks for sudo and does not require setcap.
# It may provide an external IP address other than 127.0.0.1.
# minikube tunnel
# kubectl get service/istio-ingressgateway -n istio-system
```

For local development, resolve `submarine-user-test.submarine` to IP address `127.0.0.1` by adding the following line to the file `~/etc/hosts`.

```bash
127.0.0.1 submarine-user-test.submarine # For submarine local development
```

Now we can connect to the submarine workbench at `http://submarine-user-test.submarine`.


```bash
# Step5: Stop exposing the service
# Just close the running process at Step4.

# Step6: Cleanup submarine.
kubectl delete -n submarine-user-test submarine example-submarine

# Step7: Cleanup operator.
# Just close the running process at Step2.

# Step8: Delete the submarine CRD.
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
helm dependency update ./helm-charts/submarine
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
kubectl apply -n submarine-user-test -f config/samples/_v1_submarine.yaml

# We can now expose the service like before and connect to the submarine workbench

# Step6: Cleanup submarine.
kubectl delete -n submarine-user-test submarine example-submarine

# Step7: Cleanup submarine dependencies.
helm uninstall submarine -n submarine-cloud-v3-system

# Step8: Cleanup operator.
# Note that this step deletes the namespace submarine-cloud-v3-system
make undeploy
```

### Installing Helm in a different namespace

By default, the Istio virtual service created by the operator binds to the Istio gateway `submarine-cloud-v3-system/submarine-gateway`, which is installed in the `submarine-cloud-v3-system` namespace via Helm. If `helm install` is run with `-n <your_namespace>`, set `spec.virtualserice.gateways` of your submarine CRs to `<your_namespace>/submarine-gateway`. Note that the data type is an array of strings.

### Use custom hosts

You can set the desitination hosts of the Istio virtual service by setting `spec.virtualservice.hosts` of the submarine CR. Note that the data type is an array of strings. If that field is omitted, hosts defaults to `["<namespace>.submarine"]` e.g. `["submarine-user-test.submarine"]`.

### Rebuild Operator Image

When running operator in-cluster, we need to rebuild the operator image for changes to take effect.

```bash
eval $(minikube docker-env)
make docker-build
eval $(minikube docker-env -u)
```

## Coding Style

For `go` files, please use [gofmt](https://golang.org/pkg/cmd/gofmt/) to format the code.

```bash
make fmt
```

For `yaml` files, please use [prettier](https://prettier.io/) to format the code.

## Generate API

Steps to modify custom resource definition (CRD):
1. Modify the Submarine type in `api/v1/submarine_types.go`.
2. Run `make generate` to update `api/v1/zz_generated.deepcopy.go`.
3. Run `make manifests` to update crd file `config/crd/bases/submarine.apache.org_submarines.yaml`.
4. Modify the sample submarine `config/samples/_v1_submarine`.

One can add [marker comments](https://book.kubebuilder.io/reference/markers.html) in Go code to control the manifests generation.

## Add resource

Steps to add new resource created and controlled by the operator:
1. Add or modify yaml files under `artifacts/`.
2. Modify `createSubmarine` in `contorllers/submarine_controller.go` to create resource from yaml files.
3. If needed, modify reconcile logic in `contorllers/submarine_controller.go`.
4. If needed, import new scheme in `main.go`.
5. If there are new resource types, add RBAC marker comments in `contorllers/submarine_controller.go`.
6. Run `make manifests` to update cluster role rules in `config/rbac/role.yaml`.

## Run Operator End-to-end Tests

Have the submarine operator running in-cluster or out-of-cluster before running `make test`. For verbose mode, instead run `go test ./controllers/ -v -ginkgo.v`, which outputs anything written to `GinkgoWriter` to stdout.

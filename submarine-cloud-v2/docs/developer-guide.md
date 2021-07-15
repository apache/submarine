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

## Coding Style

For `go` files, please use [gofmt](https://golang.org/pkg/cmd/gofmt/) to format the code.

For `yaml` files, please use [prettier](https://prettier.io/) to format the code.

## Generate API

**Important**: You **MUST** put this repository in a folder named `github.com/apache/`, otherwise, the code will be generated into the wrong folder. Therefore, the full path of this `developer-guide.md` should be like `SOMEWHERE_IN_FILESYSTEM/github.com/apache/submarine/submarine-cloud-v2/docs/developer-guide.md`.

We use [code-generator](https://github.com/kubernetes/code-generator) to generate a typed client, informers, listers and deep-copy functions.

Everytime when you change the codes in `submarine-cloud-v2/pkg/apis`, you must run `make api` to re-generate the API.

## Add New Dependencies

```bash
# Step1: Add the dependency to go.mod
go get ${new_dependency} # Example: go get k8s.io/code-generator

# Step2: Download the dependency to vendor/
go mod vendor
```

## Run Operator End-to-end Tests

Reference: [spark-on-k8s-operator e2e test](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/tree/master/test/e2e)

```bash
# Step1: Build image "submarine-operator" to minikube's Docker 
eval $(minikube docker-env)
make image

# Step2: Register Custom Resource Definition
kubectl apply -f artifacts/examples/crd.yaml

# Step3: Run Tests
go test ./test/e2e
```

## Run Frontend End-to-end Tests

Use the following helper script to run frontend E2E tests.

```bash
# Prerequisite: Make sure Workbench is running on $URL:$WORKBENCH_PORT.
./hack/run_frontend_e2e.sh [testcase]
```

- `[testcase]`: Check the directory [integration](../../submarine-test/test-e2e/src/test/java/org/apache/submarine/integration/).

Example:

```bash
./hack/run_frontend_e2e.sh loginIT
```

## Build Custom Images

Use the following helper script to build images and update the images used by running pods.

```
./hack/build_image.sh [all|server|database|jupyter|jupyter-gpu|mlflow]
```

Examples:

```
./hack/build_image.sh all     # build all images
./hack/build_image.sh server  # only build the server image
```

## Helm Golang API

- Function `HelmInstall` is defined in pkg/helm/helm.go.
- Example: (You can see this example in controller.go:123.)

```go
// Example: HelmInstall
// This is equal to:
// 		helm repo add k8s-as-helm https://ameijer.github.io/k8s-as-helm/
// .	helm repo update
//  	helm install helm-install-example-release k8s-as-helm/svc --set ports[0].protocol=TCP,ports[0].port=80,ports[0].targetPort=9376
// Useful Links:
//   (1) https://github.com/PrasadG193/helm-clientgo-example
// . (2) https://github.com/ameijer/k8s-as-helm/tree/master/charts/svc
helmActionConfig := helm.HelmInstall(
    "https://ameijer.github.io/k8s-as-helm/",
    "k8s-as-helm",
    "svc",
    "helm-install-example-release",
    "default",
    map[string]string {
        "set": "ports[0].protocol=TCP,ports[0].port=80,ports[0].targetPort=9376",
    },
)
// Example: HelmUninstall
// This is equal to:
//    helm uninstall helm-install-example-release
helm.HelmUninstall("helm-install-example-release", helmActionConfig)

```
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

## Create New Custom Resources

To modify the parameters of submarine resource, go to the location `submarine/submarine-cloud-v2/artifacts` to edit specific `yaml` files. In this case, you won't need to modify the code for operator.

However, if you intend to create a new custom resource, you should also add a corresponding operator code in `submarine/submarine-cloud-v2/pkg/controller`.

**Important**: Your `yaml` files should `NOT` include `helm` grammer.

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

# Step2: Install helm dependencies
helm install --wait --set dev=true submarine ../helm-charts/submarine

# Step3: Run Tests
## one can add -v to see additional logs
go test -timeout 30m ./test/e2e
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

## Rebuild Operator Image

When running operator in-cluster by Helm, we need to rebuild the operator image for changes to take effect.

```bash
eval $(minikube docker-env)
make image
eval $(minikube docker-env -u)

# Update the operator pod
helm upgrade submarine ../helm-charts/submarine
```

## Rebuild Images for Other Components

Use the following helper script to build images and update the images used by running pods.

```bash
./hack/build_image.sh [all|server|database|jupyter|jupyter-gpu]
```

Examples:

```bash
./hack/build_image.sh all     # build all images
./hack/build_image.sh server  # only build the server image
```

# Why submarine-cloud-v2?

- Because `submarine-cloud` is outdated, `submarine-cloud-v2` is the refactored version of `submarine-cloud`. In addition, after `submarine-cloud-v2` finishes, we will replace `submarine-cloud` with `submarine-cloud-v2`.

# Formatting the code

For `go` files, please use [gofmt](https://golang.org/pkg/cmd/gofmt/) to format the code.

For `yaml` files, please use [prettier](https://prettier.io/) to format the code.

# Initialization

```bash
minikube start --vm-driver=docker  --kubernetes-version v1.15.11
go mod vendor
```

# Generate API

We use the generators in [k8s.io/code-generator](https://github.com/kubernetes/code-generator) to generate a typed client, informers, listers and deep-copy functions.

**Important**: You **MUST** put this repository in a folder named `github.com/apache/`, otherwise the code will be generated into wrong folder. Therefore the full path of this `README.md` should be like `SOMEWHERE_IN_FILESYSTEM/github.com/apache/submarine/submarine-cloud-v2/README.md`.

Everytime when you change the codes in `pkg/apis`, you must run `make api` to re-generate the API.

# Add new dependencies

```bash
# Step1: Add the dependency to go.mod
go get ${new_dependency} # Example: go get k8s.io/code-generator

# Step2: Download the dependency to vendor/
go mod vendor
```

# Run Unit Test

```bash
# Step1: Register Custom Resource Definition
kubectl apply -f artifacts/examples/crd.yaml

# Step2: Create a Custom Resource
kubectl apply -f artifacts/examples/example-submarine.yaml

# Step3: Run unit test
make test-unit
```

# Run submarine-operator out-of-cluster

```bash
# Step1: Build & Run "submarine-operator"
go build -o submarine-operator
./submarine-operator

# Step2: Deploy a submarine
kubectl apply -f artifacts/examples/crd.yaml
kubectl create ns submarine-user-test
kubectl apply -n submarine-user-test -f artifacts/examples/example-submarine.yaml

# Step3: Exposing Service
# Method1 -- using minikube ip + NodePort
$ minikube ip  # you'll get the IP address of minikube, ex: 192.168.49.2

# Method2 -- using port-forwarding
$ kubectl port-forward --address 0.0.0.0 -n submarine-user-test service/traefik 32080:80

# Step4: View workbench
# http://{minikube ip}:32080(from Method1), ex: http://192.168.49.2:32080
# or http://127.0.0.1:32080 (from Method 2).

# Step5: Delete:
#   (1) Remove all relevant Helm chart releases
#   (2) Remove all resources in the namespace "submariner-user-test"
#   (3) Remove all non-namespaced resources (Ex: PersistentVolume) created by client-go API
#   (4) **Note:** The namespace "submarine-user-test" will not be deleted
kubectl delete submarine example-submarine -n submarine-user-test
```

# Run operator in-cluster

```bash
# Step1: Build image "submarine-operator" to minikube's Docker
eval $(minikube docker-env)
make image

# Step2: RBAC (ClusterRole, ClusterRoleBinding, and ServiceAccount)
kubectl apply -f artifacts/examples/submarine-operator-service-account.yaml

# Step3: Deploy a submarine-operator
kubectl apply -f artifacts/examples/submarine-operator.yaml

# Step4: Deploy a submarine
kubectl apply -f artifacts/examples/crd.yaml
kubectl create ns submarine-user-test
kubectl apply -n submarine-user-test -f artifacts/examples/example-submarine.yaml

# Step5: Inspect submarine-operator POD logs
kubectl logs -f ${submarine-operator POD}

# Step6: The operator will create a new namespace "submarine-user-test"
kubectl get all -n submarine-user-test

# Step7: Exposing Service
# Method1 -- using minikube ip + NodePort
$ minikube ip  # you'll get the IP address of minikube, ex: 192.168.49.2

# Method2 -- using port-forwarding
$ kubectl port-forward --address 0.0.0.0 -n submarine-user-test service/traefik 32080:80

# Step8: View workbench
# http://{minikube ip}:32080(from Method1), ex: http://192.168.49.2:32080
# or http://127.0.0.1:32080 (from Method 2).

# Step9: Delete:
#   (1) Remove all relevant Helm chart releases
#   (2) Remove all resources in the namespace "submariner-user-test"
#   (3) Remove all non-namespaced resources (Ex: PersistentVolume) created by client-go API
#   (4) **Note:** The namespace "submarine-user-test" will not be deleted
kubectl delete submarine example-submarine -n submarine-user-test

# Step10: Delete "submarine-operator"
kubectl delete deployment submarine-operator-demo
```

# Helm Golang API

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

- Troubleshooting:
  - If the release name exists, Helm will report the error "cannot re-use a name that is still in use".

```
helm ls
helm uninstall helm-install-example-release
```

# Build custom images when development

Use the following helper script to build images and update the images used by running pods.

```
./hack/build_image.sh [all|server|database|jupyter|jupyter-gpu|mlflow]
```

Examples:

```
./hack/build_image.sh all     # build all images
./hack/build_image.sh server  # only build the server image
```

# Run frontend E2E tests

Use the following helper script to run frontend E2E tests.

```
# Prerequisite: Make sure Workbench is running on $URL:$WORKBENCH_PORT.
./hack/run_frontend_e2e.sh [testcase]
```

- [testcase]: Check the directory [integration](../submarine-test/test-e2e/src/test/java/org/apache/submarine/integration/).

Examples:

```
./hack/run_frontend_e2e.sh loginIT
```

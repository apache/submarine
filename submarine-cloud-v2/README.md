# Why submarine-cloud-v2?
* Because `submarine-cloud` is outdated, `submarine-cloud-v2` is the refactored version of `submarine-cloud`. In addition, after `submarine-cloud-v2` finishes, we will replace `submarine-cloud` with `submarine-cloud-v2`.

# Formatting the code

For `go` files, please use [gofmt](https://golang.org/pkg/cmd/gofmt/) to format the code.

For `yaml` files, please use [prettier](https://prettier.io/) to format the code.

# Initialization
```bash
go mod vendor
chmod -R 777 vendor
```

# Generate API
* It makes use of the generators in [k8s.io/code-generator](https://github.com/kubernetes/code-generator) to generate a typed client, informers, listers and deep-copy functions. You can do this yourself using the ./update-codegen.sh script. (Note: Before you run update-codegen.sh and verify-codegen.sh, you need to move to hack/ directory at first.)
```bash
# Step1: Modify doc.go & types.go
# Step2: Generate API to pkg/generated
cd hack
./update-codegen.sh

# Step3: Verify API
./verify-codegen.sh
```

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
kubectl create ns submarine-admin
kubectl apply -n submarine-admin -f artifacts/examples/example-submarine.yaml

# Step3: "submarine-operator" will perform port-forwarding automatically.

# Step4: View workbench (127.0.0.1:32080) with your web browser

# Step5: Delete: 
#   (1) Remove all relevant Helm chart releases
#   (2) Remove all resources in the namespace "submariner-user-test"
#   (3) Remove all non-namespaced resources (Ex: PersistentVolume) created by client-go API 
#   (4) **Note:** The namespace "submarine-admin" will not be deleted
kubectl delete submarine example-submarine -n submarine-admin 
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
kubectl create ns submarine-admin
kubectl apply -n submarine-admin -f artifacts/examples/example-submarine.yaml

# Step5: Inspect submarine-operator POD logs 
kubectl logs -f ${submarine-operator POD}

# Step6: The operator will create a new namespace "submarine-user-test"
kubectl get all -n submarine-user-test 

# Step7: port-forwarding
kubectl port-forward --address 0.0.0.0 -n submarine-user-test service/traefik 32080:80

# Step8: View workbench (127.0.0.1:32080) with your web browser

# Step9: Delete: 
#   (1) Remove all relevant Helm chart releases
#   (2) Remove all resources in the namespace "submariner-user-test"
#   (3) Remove all non-namespaced resources (Ex: PersistentVolume) created by client-go API 
#   (4) **Note:** The namespace "submarine-admin" will not be deleted
kubectl delete submarine example-submarine -n submarine-admin 

# Step10: Delete "submarine-operator"
kubectl delete deployment submarine-operator-demo
```

# Helm Golang API
* Function `HelmInstall` is defined in pkg/helm/helm.go.
* Example: (You can see this example in controller.go:123.)
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
* Troubleshooting: 
  * If the release name exists, Helm will report the error "cannot re-use a name that is still in use".
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

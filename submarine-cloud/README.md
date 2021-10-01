# Submarine Operator
## Run (Method 1)
```
# Build Submarine Operator Binary
cd submarine/submarine-cloud
make build

# Create CRD (SubmarineCluster)
kubectl apply -f manifests/crd.yaml

# Create a kind cluster
./hack/kind-cluster-build.sh --name "submarine"

# Launch Submarine Operator (Method 1)
# ([Kind v0.7.0-SNAPSHOT deprecates `kind get kubeconfig-path`](https://github.com/kubernetes-sigs/cluster-api/issues/1796))
KUBECONFIG=$(kind get kubeconfig-path --name submarine)
./submarine-operator --kubeconfig=${KUBECONFIG} --alsologtostderr --v=7

# Launch Submarine Operator (Method 2)
kind get kubeconfig --name submarine > kind_kubeconfig
KUBECONFIG=$(path of kind_kubeconfig)
./bin/submarine-operator --kubeconfig=${KUBECONFIG} --alsologtostderr --v=7
```

## Run (Method 2)
```
kubectl apply -f submarine-operator
```

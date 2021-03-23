# Why submarine-cloud-v2?
* Because `submarine-cloud` is outdated, `submarine-cloud-v2` is the refactored version of `submarine-cloud`. In addition, after `submarine-cloud-v2` finishes, we will replace `submarine-cloud` with `submarine-cloud-v2`.

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
go test
```
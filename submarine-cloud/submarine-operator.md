# Submarine Operator Instructions

## Build submarine operator

### Prerequisite

1 Golang environment is required.

2 Submarine project should be in the path of ${GOPATH}/src/github.com/apache/.
Alternatively, we can create a soft link named submarine, pointing to submarine
repo, under the path of ${GOPATH}/src/github.com/apache/.

### Build submarine operator binary
```
cd ${GOPATH}/src/github.com/apache/submarine/submarine-cloud
make build
```

## Run submarine operator locally

Start a kind cluster

```
cd ${GOPATH}/src/github.com/apache/submarine/submarine-cloud/hack
./kind-cluster-build.sh --name "submarine"
```

Run submarine operator
```
# create submarine crd
kubectl apply -f ../manifests/crd.yaml
KUBECONFIG=$(kind get kubeconfig-path --name submarine)
./submarine-operator --kubeconfig=${KUBECONFIG} --alsologtostderr --v=7
OR
./submarine-operator --kubeconfig=$(kind get kubeconfig-path --name submarine) --alsologtostderr --v=7
```

## Submarine operator implementation steps

### Preparation before automatic code generation

1 Create a new directory under the samplecontroller directory and add the following files

```
[root@localhost studentcontroller]# tree pkg
pkg
├── apis
│   └── stable
│       ├── register.go
│       └── v1
│           ├── doc.go
│           ├── register.go
│           ├── types.go
Mainly prepare the declaration and registration interface of the resource object for the code generation tool
```

2 Download dependencies
```
go get -u k8s.io/apimachinery/pkg/apis/meta/v1
go get -u k8s.io/code-generator/...
go get -u k8s.io/apiextensions-apiserver/...
```

### Automatically generate Client, Informer, WorkQueue related code
```
[root@localhost launcher-k8s]# export GOPATH=/root/mygolang
[root@localhost launcher-k8s]# ./hack/update-codegen.sh
Generating deepcopy funcs
Generating clientset for stable:v1 at github.com/gfandada/samplecontroller/pkg/client/clientset
Generating listers for stable:v1 at github.com/gfandada/samplecontroller/pkg/client/listers
Generating informers for stable:v1 at github.com/gfandada/samplecontroller/pkg/client/informers
[root@localhost samplecontroller]# tree pkg
pkg
├── apis
│   └── stable
│       ├── register.go
│       └── v1
│           ├── doc.go
│           ├── register.go
│           ├── types.go
│           └── zz_generated.deepcopy.go
├── client
│   ├── clientset
│   │   └── versioned
│   │       ├── clientset.go
│   │       ├── doc.go
│   │       ├── fake
│   │       │   ├── clientset_generated.go
│   │       │   ├── doc.go
│   │       │   └── register.go
│   │       ├── scheme
│   │       │   ├── doc.go
│   │       │   └── register.go
│   │       └── typed
│   │           └── stable
│   │               └── v1
│   │                   ├── doc.go
│   │                   ├── fake
│   │                   │   ├── doc.go
│   │                   │   ├── fake_stable_client.go
│   │                   │   └── fake_student.go
│   │                   ├── generated_expansion.go
│   │                   ├── stable_client.go
│   │                   └── student.go
│   ├── informers
│   │   └── externalversions
│   │       ├── factory.go
│   │       ├── generic.go
│   │       ├── internalinterfaces
│   │       │   └── factory_interfaces.go
│   │       └── stable
│   │           ├── interface.go
│   │           └── v1
│   │               ├── interface.go
│   │               └── student.go
│   └── listers
│       └── stable
│           └── v1
│               ├── expansion_generated.go
│               └── student.go
```

### Write controller business logic

Refer to the sample-controller project, it is relatively simple to write

### Startup controller
```
[root@localhost samplecontroller]# ./samplecontroller
// This is a simple custom k8s controller, 
// used to demonstrate the idea of k8s final state operation and maintenance,
// https://github.com/gfandada/samplecontroller，

Usage:
  samplecontroller [command]

Available Commands:
  help        Help about any command
  run         run config=[kubeConfig's path]

Flags:
      --config string   config file (default is $HOME/.samplecontroller.yaml)
  -h, --help            help for samplecontroller
  -t, --toggle          Help message for toggle

Use "samplecontroller [command] --help" for more information about a command.
[root@localhost samplecontroller]# ./samplecontroller run config=/root/.kube/config 
ERROR: logging before flag.Parse: I0415 15:02:28.619121  109337 samplecontroller.go:59] Create event broadcaster
ERROR: logging before flag.Parse: I0415 15:02:28.619246  109337 samplecontroller.go:76] Listen for student's add / update / delete events
ERROR: logging before flag.Parse: I0415 15:02:28.619264  109337 samplecontroller.go:102] Start the controller business and start a cache data synchronization
ERROR: logging before flag.Parse: I0415 15:02:28.719511  109337 samplecontroller.go:107] Start 10 workers
ERROR: logging before flag.Parse: I0415 15:02:28.719547  109337 samplecontroller.go:112] all workers have been started
......

```

### Modify the crd instance file to observe the controller
```
........
kubectl apply -f test1.yaml
kubectl describe std test1
```

### Kind

```
kind load docker-image --name=submarine busybox:1.28.4
kind load docker-image --name=submarine apache/submarine:server-0.3.0-SNAPSHOT
kind load docker-image --name=submarine apache/submarine:database-0.3.0-SNAPSHOT
```

# Submarine Operator Development Instructions

See also official sample：https://github.com/kubernetes/sample-controller

# Implementation steps

## 1. Submit the CRD template and its instantiation object to the k8s cluster, so that k8s can recognize
```
1 Official document CRD 
https://kubernetes.io/docs/tasks/access-kubernetes-api/custom-resources/custom-resource-definitions/#create-a-customresourcedefinition

2 Log in to a machine that can execute kubectl commands and create a student.yaml file
[root@localhost student]# kubectl apply -f student.yaml
customresourcedefinition.apiextensions.k8s.io/students.stable.k8s.io created
[root@localhost student]# kubectl get crd
NAME                          CREATED AT
crontabs.stable.example.com   2019-03-26T01:48:32Z
students.stable.k8s.io        2019-04-12T02:42:08Z

3 Use the template student.yaml to instantiate a Student object, create test1.yaml, and similarly test2.yaml
[root@localhost student]# kubectl apply -f test1.yaml
student.stable.k8s.io/test1 created

4 kubectl describe std test1
```

## 2. Preparation before automatic code generation

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
```

## 3. Automatically generate Client, Informer, WorkQueue related code
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

## 4. Write controller business logic

Refer to the sample-controller project, it is relatively simple to write

## 5. Startup controller
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

## 6. Modify the crd instance file to observe the controller
```
........
kubectl apply -f test1.yaml
kubectl describe std test1
```

## Development
1. go mod download
Dependency packages will be automatically downloaded to `$GOPATH/pkg/mod`. Multiple projects can share cached mods.

2. go mod vendor
Copy from the mod to the vendor directory of your project so the IDE can recognize it!

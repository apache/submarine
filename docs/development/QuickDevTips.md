## 1. Introduction

This document describes some useful tips that can accelerate your development efficiency.

### Run server outside of k8s
If you are working on submarine-server, you do not need to bundle submarine-server into docker-image, and restart the helm-chart every time.

You can run each module individually and only need to re-package the submarine-server, getting rid of other unnecessary process.

1. Run db docker

```
docker run -it -p 3306:3306 -d --name submarine-database -e MYSQL_ROOT_PASSWORD=password apache/submarine:database-0.6.0-SNAPSHOT
```

2. Run k8s

```
minikube start # or other alternatives, such as kind
kubectl apply -f ./dev-support/k8s/tfjob/crd.yaml
kubectl kustomize ./dev-support/k8s/tfjob/operator | kubectl apply -f -
kubectl apply -f ./dev-support/k8s/pytorchjob/
export KUBECONFIG=/home/<user_name>/.kube/config # (in ~/.bashrc)
```
3. Package server

```
mvn clean package -DskipTests
```

4. Start server

cd submarine-dist/target/submarine-dist-0.6.0-SNAPSHOT-hadoop-2.9/submarine-dist-0.6.0-SNAPSHOT-hadoop-2.9/
./bin/submarine-daemon.sh start getMysqlJar
```
```

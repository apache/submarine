---
title: Submarine Local Deployment
slug: /
---

<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

## Prerequisite
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) 
- [helm](https://helm.sh/docs/intro/install/) (Helm v3 is minimum requirement.)
- [minikube](https://minikube.sigs.k8s.io/docs/start/).

## Deploy Kuberntes Cluster
```
$ minikube start --vm-driver=docker --cpu 8 --memory 4096 --disk-size=20G --kubernetes-versions v1.15.11
```

## Install Submarine on Kubernetes
```
$ git clone https://github.com/apache/submarine.git
$ cd submarine
$ helm install submarine ./helm-charts/submarine
```
```
NAME: submarine
LAST DEPLOYED: Fri Jan 29 05:35:36 2021
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

## Verify installation
Once you got it installed, check with below commands and you should see similar outputs:
```bash
$ kubectl get pods
```

```bash
NAME                                              READY   STATUS    RESTARTS   AGE
notebook-controller-deployment-5db8b6cbf7-k65jm   1/1     Running   0          5s
pytorch-operator-7ff5d96d59-gx7f5                 1/1     Running   0          5s
submarine-database-8d95d74f7-ntvqp                1/1     Running   0          5s
submarine-server-b6cd4787b-7bvr7                  1/1     Running   0          5s
submarine-traefik-9bb6f8577-66sx6                 1/1     Running   0          5s
tf-job-operator-7844656dd-lfgmd                   1/1     Running   0          5s
```

:::warning
Note that if you encounter below issue when installation:
:::
```bash
Error: rendered manifests contain a resource that already exists.
Unable to continue with install: existing resource conflict: namespace: , name: podgroups.scheduling.incubator.k8s.io, existing_kind: apiextensions.k8s.io/v1beta1, Kind=CustomResourceDefinition, new_kind: apiextensions.k8s.io/v1beta1, Kind=CustomResourceDefinition
```
It might be caused by the previous installed submarine charts. Fix it by running:
```bash
$ kubectl delete crd/tfjobs.kubeflow.org && kubectl delete crd/podgroups.scheduling.incubator.k8s.io && kubectl delete crd/pytorchjobs.kubeflow.org
```

## Use Port Forwarding to Access Submarine in a Cluster
```bash=
# # Listen on port 32080 on all addresses, forwarding to 80 in the pod
$ kubectl port-forward --address 0.0.0.0 service/submarine-traefik 32080:80
```
## Open Workbench in the browser.
Open http://127.0.0.1:32080. The default username and password is `admin` and `admin`

![](https://i.imgur.com/DkZhyEG.png)

## Uninstall Submarine
```bash
$ helm delete submarine
```
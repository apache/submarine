---
title: Setup a Kubernetes cluster using Kind
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

## Create K8s cluster
We recommend using [`kind`](https://kind.sigs.k8s.io/) to setup a Kubernetes cluster on a local machine.

Running the following command:
```
kind create cluster --image kindest/node:v1.15.6 --name submarine
kubectl create namespace submarine
```

## Kubernetes Dashboard (optional)

### Deploy
To deploy Dashboard, execute following command:
```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0-beta8/aio/deploy/recommended.yaml
```

### Create RBAC
Ensure to grant the cluster access permission of dashboard, run the following command:
```
kubectl create serviceaccount dashboard-admin-sa
kubectl create clusterrolebinding dashboard-admin-sa --clusterrole=cluster-admin --serviceaccount=default:dashboard-admin-sa
```

### Get access token (optional)
If you want to use the token to login the dashboard, run the following command to get key:
```
kubectl get secrets
# select the right dashboard-admin-sa-token to describe the secret
kubectl describe secret dashboard-admin-sa-token-6nhkx
```

### Start dashboard service
To start the dashboard service, we can run the following command:
```
kubectl proxy
```

Now access Dashboard at:
> http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/

---
title: Setup a Kubernetes cluster using KinD
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
We recommend using [`KinD`](https://kind.sigs.k8s.io/) to setup a Kubernetes cluster on a local machine.

Running the following command, and specify the KinD version and Kubernetes version [`here`](../devDocs/Dependencies).
```bash
# Download the specific version of KinD (must >= v0.6.0)
export KIND_VERSION=v0.11.1
curl -Lo ./kind https://github.com/kubernetes-sigs/kind/releases/download/${KIND_VERSION}/kind-linux-amd64
# Make the binary executable
chmod +x ./kind
# Move the binary to your executable path
sudo mv ./kind /usr/local/bin/
# Create cluster with specific version of kubernetes
export KUBE_VERSION=v1.15.12
kind create cluster --image kindest/node:${KUBE_VERSION}
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

Dashboard screenshot:

![](../assets/kind-dashboard.png)

---
title: Custom Configuation
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
## Helm Chart Volume Type
Submarine can support various [volume types](https://kubernetes.io/docs/concepts/storage/volumes/#nfs), currently including hostPath (default) and NFS. It can be easily configured in the `./helm-charts/submarine/values.yaml`, or you can override the default values in `values.yaml` by [helm CLI](https://helm.sh/docs/helm/helm_install/).

#### hostPath
- In hostPath, you can store data directly in your node.
- Usage:
  1. Configure setting in `./helm-charts/submarine/values.yaml`.
  2. To enable hostPath storage, set `.storage.type` to `host`.
  3. To set the root path for your storage, set `.storage.host.root` to `<any-path>`
- Example:
  ```yaml
  # ./helm-charts/submarine/values.yaml
  storage:
    type: host
    host:
      root: /tmp
  ```


#### NFS (Network File System)
- In NFS, it allows multiple clients to access a shared space.
- Prerequisite:
  1. A pre-existing NFS server. You have two options.
      1. Create NFS server
          ```bash
          kubectl create -f ./dev-support/nfs-server/nfs-server.yaml
          ```
          It will create a nfs-server pod in kubernetes cluster, and expose nfs-server ip at `10.96.0.2`
      2. Use your own NFS server
  2. Install NFS dependencies in your nodes
      - Ubuntu
          ```bash
          apt-get install -y nfs-common
          ```
      - CentOS
          ```bash
          yum install nfs-util
          ```
- Usage:
  1. Configure setting in `./helm-charts/submarine/values.yaml`.
  2. To enable NFS storage, set `.storage.type` to `nfs`.
  3. To set the ip for NFS server, set `.storage.nfs.ip` to `<any-ip>`
- Example:
  ```yaml
  # ./helm-charts/submarine/values.yaml
  storage:
    type: nfs
    nfs:
      ip: 10.96.0.2
  ```
### Access to Submarine Server
Submarine server by default expose 8080 port within K8s cluster. After Submarine v0.5
uses Traefik as reverse-proxy by default. If you don't want to
use Traefik, you can modify below value to ***false*** in `./helm-charts/submarine/values.yaml`.
```yaml
# Use Traefik by default
traefik:
  enabled: true
```

To access the server from outside of the cluster, we use Traefik ingress controller and
NodePort for external access.\
Please refer to `./helm-charts/submarine/charts/traefik/values.yaml` and [Traefik docs](https://docs.traefik.io/)
for more details if you want to customize the default value for Traefik.


*Notice:*
If you use `kind` to run local Kubernetes cluster,
please refer to this [docs](https://kind.sigs.k8s.io/docs/user/configuration/#extra-port-mappings)
and set the configuration "extraPortMappings" when creating the k8s cluster.

```
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 32080
    hostPort: [the port you want to access]
```


```
# Use nodePort and Traefik ingress controller by default.
# To access the submarine server, open the following URL in your browser.
http://127.0.0.1:32080
```


If minikube is installed, use the following command to find the URL to the Submarine server.
```
$ minikube service submarine-traefik --url
```

## Kubernetes Dashboard (optional)

### Deploy
To deploy Dashboard, execute the following command:
```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0-beta8/aio/deploy/recommended.yaml
```

### Create RBAC
Run the following commands to grant the cluster access permission of dashboard:
```
kubectl create serviceaccount dashboard-admin-sa
kubectl create clusterrolebinding dashboard-admin-sa --clusterrole=cluster-admin --serviceaccount=default:dashboard-admin-sa
```

### Get access token (optional)
If you want to use the token to login the dashboard, run the following commands to get key:
```
kubectl get secrets
# select the right dashboard-admin-sa-token to describe the secret
kubectl describe secret dashboard-admin-sa-token-6nhkx
```

### Start dashboard service
```
kubectl proxy
```

Now access Dashboard at:
> http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/

Dashboard screenshot:

![](/img/kind-dashboard.png)

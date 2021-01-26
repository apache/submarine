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


# Deploy Submarine On K8s

## Deploy Submarine Using Helm Chart (Recommended)

Submarine's Helm Chart will deploy Submarine Server, TF/PyTorch Operator, Notebook controller
and Traefik. We use the TF/PyTorch operator to run tf/pytorch job, the notebook controller to
manage jupyter notebook and Traefik as reverse-proxy.


### Install Helm

Helm v3 is minimum requirement.
See here for installation: https://helm.sh/docs/intro/install/

### Install Submarine

The Submarine helm charts is released with the source code for now.
Please go to `http://submarine.apache.org/download.html` to download

- Install Helm charts from source code
```bash
cd <PathTo>/submarine
helm install submarine ./helm-charts/submarine
```
This will install submarine in the "default" namespace.
The images are from Docker hub `apache/submarine`. See `./helm-charts/submarine/values.yaml` for more details

If we'd like use a different namespace like "submarine"
```bash
kubectl create namespace submarine
helm install submarine ./helm-charts/submarine -n submarine
```

> Note that if you encounter below issue when installation:
```bash
Error: rendered manifests contain a resource that already exists.
Unable to continue with install: existing resource conflict: namespace: , name: podgroups.scheduling.incubator.k8s.io, existing_kind: apiextensions.k8s.io/v1beta1, Kind=CustomResourceDefinition, new_kind: apiextensions.k8s.io/v1beta1, Kind=CustomResourceDefinition
```
It might be caused by the previous installed submarine charts. Fix it by running:
```bash
kubectl delete crd/tfjobs.kubeflow.org && kubectl delete crd/podgroups.scheduling.incubator.k8s.io && kubectl delete crd/pytorchjobs.kubeflow.org
```

- Verify installation

Once you got it installed, check with below commands and you should see similar outputs:
```bash
kubectl get pods
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

### Configure volume type
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

```
# Use nodePort and Traefik ingress controller by default.
# To access the submarine server, open the following URL in your browser.
http://127.0.0.1:32080
```

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

### Uninstall Submarine
```bash
helm delete submarine
```

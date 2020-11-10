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

Or you can use port-forward to forward a local port to a port on the
submarine server pod.

```bash
# Use port-forward
kubectl port-forward svc/submarine-server 8080:8080

# In another terminal. Run below command to verify it works
curl http://127.0.0.1:8080/api/v1/experiment/ping
{"status":"OK","code":200,"success":true,"message":null,"result":"Pong","attributes":{}}
```

### Uninstall Submarine
```bash
helm delete submarine
```

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

Submarine's Helm Chart will not only deploy Submarine Server, but also deploys TF Operator / PyTorch Operator (which will be used by Submarine Server to run TF/PyTorch jobs on K8s).


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
NAME                                 READY     STATUS    RESTARTS   AGE
pytorch-operator-54854bf847-x65nk    1/1       Running   0          5m
submarine-database-5f74f747d-dzmf6   1/1       Running   0          5m
submarine-server-6f449bc967-cqkkv    1/1       Running   0          5m
tf-job-operator-c9cd7ccbd-4dzcs      1/1       Running   0          5m
```

### Enable local access to Submarine Server

```bash
kubectl port-forward svc/submarine-server 8080:8080

# In another terminal. Run below command to verify it works
curl http://127.0.0.1:8080/api/v1/experiment/ping
{"status":"OK","code":200,"success":true,"message":null,"result":"Pong","attributes":{}}
```

### Uninstall Submarine
```bash
helm delete submarine
```

### Create Your Custom Submarine Images (Optional)
Sometimes we'd like to do some modifications on the images.
After that, you need to rebuild submarine images:
> Note that you need to make sure the images built above can be accessed in k8s
> Usually this needs a rename and push to a proper Docker registry.

```bash
mvn clean package -DskipTests
```

Build submarine server image:
```bash
./dev-support/docker-images/submarine/build.sh
```

Build submarine database image:
```bash
./dev-support/docker-images/database/build.sh
```

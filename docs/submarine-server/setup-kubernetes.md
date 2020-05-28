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

# Deploy Submarine Server on Kubernetes
This guide covers the deployment server on kubernetes cluster.

## Experiment environment

### Setup Kubernetes
We recommend using [`kind`](https://kind.sigs.k8s.io/) to setup a Kubernetes cluster on a local machine.

Running the following command:
```
kind create cluster --image kindest/node:v1.15.6 --name k8s-submarine
kubectl create namespace submarine
```

### Kubernetes Dashboard (optional)

#### Deploy
To deploy Dashboard, execute following command:
```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0-beta8/aio/deploy/recommended.yaml
```

#### Create RBAC
Ensure to grant the cluster access permission of dashboard, run the following command:
```
kubectl create serviceaccount dashboard-admin-sa
kubectl create clusterrolebinding dashboard-admin-sa --clusterrole=cluster-admin --serviceaccount=default:dashboard-admin-sa
```

#### Get access token (optional)
If you want to use the token to login the dashboard, run the following command to get key:
```
kubectl get secrets
# select the right dashboard-admin-sa-token to describe the secret
kubectl describe secret dashboard-admin-sa-token-6nhkx
```

#### Access
To start the dashboard service, we can run the following command:
```
kubectl proxy
```

Now access Dashboard at:
> http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/

### Setup Submarine

#### Local

##### Get package
You can dowload submarine from releases page or build from source.

##### Configuration
Copy the kube config into `conf/k8s/config` or modify the `conf/submarine-site.xml`:
```
<property>
  <name>submarine.k8s.kube.config</name>
  <value>PATH_TO_KUBE_CONFIG</value>
</property>
```

##### Start Submarine Server
Running the submarine server, executing the following command:
```
# if build from source. You need to run this under the target dir like submarine-dist/target/submarine-dist-0.4.0-SNAPSHOT-hadoop-2.9/submarine-dist-0.4.0-SNAPSHOT-hadoop-2.9/
./bin/submarine-daemon.sh start getMysqlJar
```

The REST API URL is: `http://127.0.0.1:8080/api/v1/jobs`

#### Deploy Tensorflow Operator
For more info see [deploy tensorflow operator](./ml-frameworks/tensorflow.md).

#### Deploy PyTorch Operator
```bash
cd <submarine_code_path_root>/dev-support/k8s/pytorchjob
./deploy-pytorch-operator.sh

```


### Use Helm Chart to deploy

#### Create images
submarine server
```bash
./dev-support/docker-images/submarine/build.sh
```

submarine database
```bash
./dev-support/docker-images/database/build.sh
```

#### install helm
For more info see https://helm.sh/docs/intro/install/

#### Deploy submarine server, mysql
You can modify some settings in ./helm-charts/submarine/values.yaml
```bash
helm install submarine ./helm-charts/submarine
```

#### Delete deployment
```bash
helm delete submarine 
```

#### port-forward {host port}:{container port}
```bash
kubectl port-forward svc/submarine-server 8080:8080 --address 0.0.0.0
```

## Production environment

### Setup Kubernetes
For more info see https://kubernetes.io/docs/setup/#production-environment

### Setup Submarine
It's will come soon.

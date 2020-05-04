<!--
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-->
# Deploy Jupyter Notebook on Kubernetes
This guide covers the deployment Jupyter Notebook on kubernetes cluster.

## Experiment environment
### Setup Kubernetes
We recommend using [kind](https://kind.sigs.k8s.io/) to setup a Kubernetes cluster on a local machine.

You can use Extra mounts to mount your host path to kind node and use Extra port mappings to port
forward to the kind nodes. Please refer to [kind configuration](https://kind.sigs.k8s.io/docs/user/configuration/#extra-mounts)
for more details.

You need to create a kind config file. The following is an example :
```
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraMounts:
  # add a mount from /path/to/my/files on the host to /files on the node
  - hostPath: /tmp/jovyan
    containerPath: /home/jovyan
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  # exposing additional ports to be used for NodePort services
  - containerPort: 30070
    hostPort: 8888
    protocol: TCP
```

Running the following command:

```
kind create cluster --image kindest/node:v1.15.6 --config <path-to-kind-config> --name k8s-submarine
kubectl create namespace submarine
```

### Deploy Jupyter Notebook
Once you have a running Kubernetes cluster, you can write a YAML file to deploy a jupyter notebook.
In this [example yaml](./jupyter.yaml), we use [jupyter/minimal-notebook](https://hub.docker.com/r/jupyter/minimal-notebook/)
to make a single notebook running on the kind node.

```
kubectl apply -f jupyter.yaml --namespace submarine
```

Once jupyter notebook is running, you can access the notebook server from the browser using http://localhost:8888 on local machine.

You can enter and store a password for your notebook server with:
```
kubectl exec -it <jupyter-pod-name> -- jupyter notebook password
```
After restarting the notebook server,  you can login jupyter notebook with your new password.

If you want to use JupyterLab :
```
http://localhost:8888/lab
```

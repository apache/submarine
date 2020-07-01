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

# Notebook Guide
This guide describes how to use Kubeflow's notebook-controller to manage jupyter notebook instances.

## Notebook Controller
The controller creates a StatefulSet to manage the notebook instance, and a Service to expose its port. \
Please refer to the [link](https://github.com/kubeflow/kubeflow/tree/master/components/notebook-controller) for more info.


### Pod Spec
To specify the PodSpec for the jupyter notebook.
```yaml
apiVersion: kubeflow.org/v1alpha1
kind: Notebook
metadata:
  name: {NOTEBOOK_NAME}
  namespace: {NAMESPACE}
  labels:
      app: {NOTEBOOK_NAME}
spec:
  template:
    spec:
      containers:
      - image: {IMAGE_NAME}
        name: {NOTEBOOK_NAME}
        env: []
        resources:
          requests:
            cpu: "0.5"
            memory: "1.0Gi"
      volumes: []
        ...
        ..
```
You could refer to this sample [Dockerfile](../../../dev-support/docker-images/jupyter/Dockerfile) for building your own
jupyter docker image and the CR (Notebook) [example](jupyter-example.yaml).

### Create a notebook instance
```
kubectl apply -f jupyter-example.yaml
```

## Access the notebook locally
The controller creates a Service which will target TCP port 8888 on jupyter pod and be exposed on port 80 internally. \
You can use the following command to set up port forwarding to the notebook.
```
kubectl port-forward -n ${NAMESPACE} svc/${NOTEBOOK_NAME} 8888:80
```
To access the jupyter notebook, open the following URL in your browser.
```
http://localhost:8888/notebook/${NAMESPACE}/${NOTEBOOK_NAME}
```

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

# Deploy PyTorch Operator on Kubernetes

## PyTorchJob
We support PyTorch job on kubernetes by using the pytorch-operator as a runtime. For more info about tf-operator see [here](https://github.com/kubeflow/pytorch-operator).

### Deploy pytorch-operator
> If you don't have the `submarine` namespace on your K8s cluster, you should create it first. Run command: `kubectl create namespace submarine`

Running the follow commands:
```
kubectl apply -f ./dev-support/k8s/pytorchjob/
```


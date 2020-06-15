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

# Submarine on K8s
Submarine for K8s supports distributed TensorFlow and PyTorch.

Submarine can run on K8s >= 1.14, supports features like GPU isolation.

## Install Submarine
Submarine can be deployed on any K8s environment if version matches. If you don't have a running K8s, you can follow the steps to set up a K8s using [kind, Kubernetes-in-Docker](https://kind.sigs.k8s.io/) for testing purpose, we provides simple [tutorial](kind.md).

### Use Helm Charts
After you have an up-and-running K8s, you can follow [Submarine Helm Charts Guide](helm.md) to deploy Submarine services on K8s cluster in minutes.

## Use Submarine

### Model training (experiment) on K8s
- [Run model training using Tensorflow](run-tensorflow-experiment.md)
- [Run model training using PyTorch](run-pytorch-experiment.md)
- [Experiment API Reference](api/experiment.md)


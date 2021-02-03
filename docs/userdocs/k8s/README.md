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

We have validated Submarine on the following versions:

| K8s Version   | Support?  |
| ------------- |:-------------:|
| 1.13.x (or earlier) | X |
| 1.14.x | √ |
| 1.15.x | √ |
| 1.16.x | √ |
| 1.17.x | To be verified |
| 1.18.x | To be verified |

## Install Submarine

### Setup Kubernetes
Submarine can be deployed on any K8s environment if version matches. If you don't have a running K8s, you can set up a K8s using [Docker Desktop](https://www.docker.com/products/docker-desktop), [MiniKube](https://kubernetes.io/docs/tasks/tools/install-minikube/), or [kind, Kubernetes-in-Docker](https://kind.sigs.k8s.io/).

From our experiences, Docker Desktop is an easier choice.

### Install Submarine Use Helm Charts
After you have an up-and-running K8s, you can follow [Submarine Helm Charts Guide](helm.md) to deploy Submarine services on K8s cluster in minutes.

## Using Jupyter notebooks in Submarine

Please refer to [Notebook guide](notebook.md) if you want to use jupyter notebook in Submarine.

## Model Training (Experiment) on K8s

### Model Training with Submarine UI

You can now submit Submarine model training on the UI, please refer to [Submit Experiment from Submarine UI](./run-tensorflow-experiment-ui.md)

### Model Training With Submarine Python SDK

You can also follow the guidance to run Submarine model training using Python SDK, please refer to [Submit Experiment using Python SDK](./run-experiment-python-sdk.md)

### Model Training With Submarine REST API

Alternatively, we support use REST API to submit, list, delete experiments (model training)

- [Run model training using Tensorflow](run-tensorflow-experiment-rest.md)
- [Run model training using PyTorch](run-pytorch-experiment-rest.md)
- [Experiment API Reference](api/experiment.md)


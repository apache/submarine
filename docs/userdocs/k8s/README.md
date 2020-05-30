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

# Running Submarine on K8s

Submarine for K8s supports standalone distributed TensorFlow and PyTorch.

Submarine can run on K8s >= (FIXME, version), supports features like GPU isolation.

## Submarine on K8s guide

### Prepare K8s and deploy Submarine Service

[Setup Kubernetes](setup-kubernetes.md): Submarine can be deployed on any K8s environment if version matches. If you don't have a running K8s, you can follow the steps to set up a K8s using [kind, Kubernetes-in-Docker](https://kind.sigs.k8s.io/) for testing purpose.

After you have an up-and-running K8s, you can follow [Deploy Submarine Services on K8s](deploy-submarine.md) guide to deploy Submarine services on K8s using Helmchart in minutes (FIXME: is it true?).

### Use Submarine

#### Model training (experiment) on K8s

- [Run model training using Tensorflow](run-tensorflow-on-k8s.md)
- [Run model training using PyTorch](FIXME, add one).

## References

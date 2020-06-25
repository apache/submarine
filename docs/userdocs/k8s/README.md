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

[FIXME]: is it accurate?

| K8s Version   | Support?  |
| ------------- |:-------------:|
| 1.13.x (or earlier) | X |
| 1.14.x | √ |
| 1.15.x | √ |
| 1.16.x | √ |
| 1.17.x | To be verified |
| 1.17.x | To be verified |

## Install Submarine

### Setup Kubernetes
Submarine can be deployed on any K8s environment if version matches. If you don't have a running K8s, you can set up a K8s using [Docker Desktop](https://www.docker.com/products/docker-desktop), [MiniKube](https://kubernetes.io/docs/tasks/tools/install-minikube/), or [kind, Kubernetes-in-Docker](https://kind.sigs.k8s.io/).

From our experiences, Docker Desktop is an easier choice.

### Install Submarine Use Helm Charts
After you have an up-and-running K8s, you can follow [Submarine Helm Charts Guide](helm.md) to deploy Submarine services on K8s cluster in minutes.

## Model Training (Experiment) on K8s

### Model Training With Submarine Python SDK

Submarine Python SDK can runs on any machine and it will talk to Submarine Server via REST API. So you can install Submarine Python SDK on your laptop, a gateway machine, your favorite IDE (like PyCharm/Jupyter, etc.).

#### Prepare Python Environment to run Submarine SDK

First of all

Submarine SDK requires Python3.7+.
It's better to use a new Python environment created by `Anoconda` or Python `virtualenv` to try this to avoid trouble to existing Python environment.
A sample Python virtual env can be setup like this:
```bash
wget https://files.pythonhosted.org/packages/33/bc/fa0b5347139cd9564f0d44ebd2b147ac97c36b2403943dbee8a25fd74012/virtualenv-16.0.0.tar.gz
tar xf virtualenv-16.0.0.tar.gz

# Make sure to install using Python 3
python3 virtualenv-16.0.0/virtualenv.py venv
. venv/bin/activate
```

#### Install Submarine SDK

**Install SDK from pypi.org (recommended)**

Starting from 0.4.0, Submarine provides Python SDK. Please change it to a proper version needed.

```bash
pip install submarine-sdk==0.4.0
```

**Install SDK from source code**

Please first clone code from github or go to `http://submarine.apache.org/download.html` to download released source code.

```bash
git clone https://github.com/apache/submarine.git
git checkout <correct release tag/branch>
cd submarine/submarine-sdk/pysubmarine
pip install .
```

#### Run with Submarine Python SDK

Assuming you've installed submarine on K8s and forward the service to localhost, now you can open a Python shell, Jupyter notebook or any tools with Submarine SDK installed.

Follow [SDK experiment example](../../../submarine-sdk/pysubmarine/example/submarine_experiment_sdk.ipynb) to try the SDK.

### Model Training With Submarine REST API

Alternatively, we support use REST API to submit, list, delete experiments (model training)

- [Run model training using Tensorflow](run-tensorflow-experiment.md)
- [Run model training using PyTorch](run-pytorch-experiment.md)
- [Experiment API Reference](api/experiment.md)


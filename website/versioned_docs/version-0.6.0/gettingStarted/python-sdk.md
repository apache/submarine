---
title: Submarine Python SDK
---

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

Submarine Python SDK can runs on any machine and it will talk to Submarine Server via REST API. So you can install Submarine Python SDK on your laptop, a gateway machine, your favorite IDE (like PyCharm/Jupyter, etc.).

Furthermore, Submarine supports an extensible package of CTR models based on **TensorFlow** and **PyTorch** along with lots of core components layers that can be used to easily build custom models. You can train any model with `model.train()` and `model.predict()`.

## Prepare Python Environment to run Submarine SDK

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

## Install Submarine SDK

### Install SDK from pypi.org (recommended)

Starting from `0.4.0`, Submarine provides Python SDK. Please change it to a proper version needed. 

More detail: https://pypi.org/project/apache-submarine/

```bash
# Install latest stable version
pip install apache-submarine
# Install specific version
pip install apache-submarine==<REPLACE_VERSION>
```

### Install SDK from source code

Please first clone code from github or go to `http://submarine.apache.org/download.html` to download released source code.

```bash
git clone https://github.com/apache/submarine.git
# (optional) chackout specific branch or release
git checkout <correct release tag/branch>
cd submarine/submarine-sdk/pysubmarine
pip install .
```

## Manage Submarine Experiment

Assuming you've installed submarine on K8s and forward the traefik service to localhost, now you can open a Python shell, Jupyter notebook or any tools with Submarine SDK installed.

Follow [SDK experiment example](https://github.com/apache/submarine/blob/master/submarine-sdk/pysubmarine/example/submarine_experiment_sdk.ipynb) to run an experiment.

## Training a DeepFM model 
The Submarine also supports users to train an easy-to-use CTR model with a few lines of code and a configuration file, so they don’t need to reimplement the model by themself. In addition, they can train the model on both local on distributed systems, such as Hadoop or Kubernetes.


Follow [SDK DeepFM example](https://github.com/apache/submarine/blob/master/submarine-sdk/pysubmarine/example/deepfm_example.ipynb) to try the model.
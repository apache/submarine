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

# Generic Job Spec

## Motivation
As the machine learning platform, the submarine should support multiple machine learning frameworks, such as Tensorflow, Pytorch etc. But different framework has different distributed components for the training job. So that we designed a generic job spec to abstract the training job across different frameworks. In this way, the submarine-server can hide the complexity of underlying infrastructure differences and provide a cleaner interface to manager jobs

## Proposal
Considering the Tensorflow and Pytorch framework, we propose one spec which consists of library spec, submitter spec and task specs etc. Such as:
```yaml
name: "mnist"
librarySpec:
  name: "TensorFlow"
  version: "2.1.0"
  image: "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0"
  cmd: "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150"
  envVars:
    ENV_1: "ENV1"
submitterSpec:
  type: "k8s"
  namespace: "submarine"
taskSpecs:
  Ps:
    name: tensorflow
    replicas: 2
    resources: "cpu=4,memory=2048M,nvidia.com/gpu=1"
  Worker:
    name: tensorflow
    replicas: 2
    resources: "cpu=4,memory=2048M,nvidia.com/gpu=1"
```

### Library Spec
The library spec describes the info about machine learning framework. All the fields as below:

| field | type | optional | description |
|---|---|---|---|
| name | string | NO | Machine Learning Framework name. Only `"tensorflow"` and `"pytorch"` is supported. It doesn't matter if the value is uppercase or lowercase.|
| version | string | NO | The version of ML framework. Such as: 2.1.0 |
| image | string | NO | The public image used for each task if not specified. Such as: apache/submarine |
| cmd | string | YES | The public entry cmd for the task if not specified. |
| envVars | key/value | YES | The public env vars for the task if not specified. |

### Submitter Spec
It describes the info of submitter which the user spcified, such as yarn, yarnservice or k8s. All the fields as below:

| field | type | optional | description |
|---|---|---|---|
| type | string | NO | The submitter type, supports `k8s` now |
| configPath | string | YES | The config path of the specified resource manager. You can set it in submarine-site.xml if run submarine-server locally |
| namespace | string | NO | It's known as queue in Apache Hadoop YARN and namespace in Kubernetes. |
| kind | string | YES | It's used for k8s submitter, supports TFJob and PyTorchJob |
| apiVersion | string | YES | It should pair with the kind, such as the TFJob's api version is `kubeflow.org/v1` |

### Task Spec
It describes the task info, the tasks make up the job. So it must be specified when submit the job. All the tasks should putted into the key value collection. Such as:
```yaml
taskSpecs:
  Ps:
    name: tensorflow
    replicas: 2
    resources: "cpu=4,memory=2048M,nvidia.com/gpu=1"
  Worker:
    name: tensorflow
    replicas: 2
    resources: "cpu=4,memory=2048M,nvidia.com/gpu=1"
```

All the fileds as below:

| field | type | optional | description |
|---|---|---|---|
| name | string | YES | The job task name, if not specify using the library name |
| image | string | YES | The job task image |
| cmd | string | YES | The entry command for running task |
| envVars | key/value | YES | The env vars for the task |
| resources | string | NO | The limit resource for the task. Formatter: cpu=%s,memory=%s,nvidia.com/gpu=%s |

## Implements
For more info see [SUBMARINE-321](https://issues.apache.org/jira/browse/SUBMARINE-321)

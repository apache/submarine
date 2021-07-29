---
title: Experiment REST API
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

## Create Experiment (Using Anonymous/Embedded Environment)

```sh
POST /api/v1/experiment
```

### Parameters

Put ExperimentSpec in request body.

#### **ExperimentSpec**

| Field Name  | Type                              | Description                             |
| ----------- | --------------------------------- | --------------------------------------- |
| meta        | ExperimentMeta                    | Meta data of the experiment template.   |
| environment | EnvironmentSpec                   | Environment of the experiment template. |
| spec        | Map<String, ExperimentTaskSpec> | Spec of pods.                           |
| code        | CodeSpec                          | Experiment codespec.                    |

#### **ExperimentMeta**

| Field Name | Type                  | Description              |
| ---------- | --------------------- | ------------------------ |
| name       | String                | Experiment name.         |
| namespace  | String                | Experiment namespace.    |
| framework  | String                | Experiemnt framework.    |
| cmd        | String                | Command.                 |
| envVars    | Map<String, String>   | Environmental variables. |

#### **EnvironmentSpec**

There are two types of environment: Anonymous and Predefined.
- Anonymous environment: only specify `dockerImage` in environment spec. The container will be built on the docker image.
- Embedded environment: specify `name` in environment spec. The container will be built on the existing environment (including dockerImage and kernalSpec).

See more details in [environment api](https://submarine.apache.org/docs/userDocs/api/environment).

#### **ExperimentTaskSpec**

| Field Name | Type                  | Description              |
| ---------- | --------------------- | ------------------------ |
| replicas   | Integer               | Numbers of replicas.     |
| resoureces | String                | Resouces of the task     |
| name       | String                | Task name.               |
| image      | String                | Image name.              |
| cmd        | String                | Command.                 |
| envVars    | Map<String, String>   | Environmental variables. |

#### **CodeSpec**

Currently only support pulling from github. HDFS, NFS and s3 are in development

| Field Name | Type                          | Description             |
| ---------- | ------------------------------| ----------------------- |
| syncMode   | String \(git\|hdfs\|nfs\|s3\) | sync mode of code spec. |
| url        | String                        | url of code spec.       |

### Code Example

**shell**

```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "meta": {
    "name": "tf-mnist-json",
    "namespace": "default",
    "framework": "TensorFlow",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "environment": {
    "image": "apache/submarine:tf-mnist-with-summaries-1.0"
  },
  "spec": {
    "Ps": {
      "replicas": 1,
      "resources": "cpu=1,memory=1024M"
    },
    "Worker": {
      "replicas": 1,
      "resources": "cpu=1,memory=2048M"
    }
  }
}
' http://127.0.0.1:32080/api/v1/experiment
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment_1626160071451_0002",
    "name":"tf-mnist-json",
    "uid":"5a6ec922-6c90-43d4-844f-039f6804ed36",
    "status":"Accepted",
    "acceptedTime":"2021-07-13T16:47:51.000+08:00",
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
    "spec":{
      "meta":{
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{"ENV_1":"ENV1"}
      },
      "environment":{
        "name":null,
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":"apache/submarine:tf-mnist-with-summaries-1.0"
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"1024M","cpu":"1"}
        },
        "Worker":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"2048M","cpu":"1"}
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

## Create Experiment (Using Pre-defined/Stored Environment)

```
POST /api/v1/experiment
```

### Parameters

Put ExperimentSpec in request body.

### Code Example

**shell**

```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "meta": {
    "name": "tf-mnist-json",
    "namespace": "default",
    "framework": "TensorFlow",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "environment": {
    "name": "my-submarine-env"
  },
  "spec": {
    "Ps": {
      "replicas": 1,
      "resources": "cpu=1,memory=1024M"
    },
    "Worker": {
      "replicas": 1,
      "resources": "cpu=1,memory=2048M"
    }
  }
}
' http://127.0.0.1:32080/api/v1/experiment
```
Above example assume environment "my-submarine-env" already exists in Submarine. Please refer Environment API Reference doc to [environment rest api](https://submarine.apache.org/docs/userDocs/api/environment).

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment_1626160071451_0005",
    "name":"tf-mnist-json",
    "uid":"4944c603-0f21-49e5-826a-2ff820bb4d93",
    "status":"Accepted",
    "acceptedTime":"2021-07-13T16:57:27.000+08:00",
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
    "spec":{
      "meta":{
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{"ENV_1":"ENV1"}
      },
      "environment":{
        "name":"my-submarine-env",
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":null
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"1024M","cpu":"1"}
        },
        "Worker":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"2048M","cpu":"1"}
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

## List Experiment

```
GET /api/v1/experiment
```

### Code Example

**shell**

```sh
curl -X GET http://127.0.0.1:32080/api/v1/experiment
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":
  [{
    "experimentId":"experiment_1626160071451_0001",
    "name":"newexperiment1",
    "uid":"b895985c-411c-4e89-90e0-c60a2a8a4235",
    "status":"Succeeded",
    "acceptedTime":"2021-07-13T16:21:31.000+08:00",
    "createdTime":"2021-07-13T16:21:31.000+08:00",
    "runningTime":"2021-07-13T16:21:46.000+08:00",
    "finishedTime":"2021-07-13T16:26:54.000+08:00",
    "spec":{
      "meta":{
        "name":"newexperiment1",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{"ENV1":"ENV1"}
      },
      "environment":{
        "name":null,
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":"apache/submarine:tf-mnist-with-summaries-1.0"
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"1024M","cpu":"1"}
        },
        "Worker":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"1024M","cpu":"1"}
        }
      },
      "code":null
    }
  },
  {
    "experimentId":"experiment_1626160071451_0005",
    "name":"tf-mnist-json",
    "uid":"4944c603-0f21-49e5-826a-2ff820bb4d93",
    "status":"Accepted",
    "acceptedTime":"2021-07-13T16:57:27.000+08:00",
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
    "spec":{
      "meta":{
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{"ENV_1":"ENV1"}
      },
      "environment":{
        "name":"my-submarine-env",
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":null
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"1024M","cpu":"1"}
        },
        "Worker":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"2048M","cpu":"1"}
        }
      },
      "code":null
    }
  }],
  "attributes":{}
}
```

## Get Experiment

```
GET /api/v1/experiment/{id}
```

### Parameters

| Field Name | Type   | In   | Description    |
| ---------- | ------ | ---- | -------------- |
| id         | String | path | Experiment id. |
### Code Example

**shell**

```sh
curl -X GET http://127.0.0.1:32080/api/v1/experiment/experiment_1626160071451_0005
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment_1626160071451_0005",
    "name":"tf-mnist-json",
    "uid":"4944c603-0f21-49e5-826a-2ff820bb4d93",
    "status":"Accepted",
    "acceptedTime":"2021-07-13T16:57:27.000+08:00",
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
    "spec":{
      "meta":{
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{"ENV_1":"ENV1"}
      },
      "environment":{
        "name":"my-submarine-env",
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":null
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"1024M","cpu":"1"}
        },
        "Worker":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"2048M","cpu":"1"}
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

## Patch Experiment

```
PATCH /api/v1/experiment/{id}
```

### Parameters

| Field Name  | Type                              | In   | Description                             |
| ----------- | --------------------------------- | ---- | --------------------------------------- |
| id          | String                            | path | Experiment id.                          |
| meta        | ExperimentMeta                    | body | Meta data of the experiment template.   |
| environment | EnvironmentSpec                   | body | Environment of the experiment template. |
| spec        | Map<String, ExperimentTaskSpec>   | body | Spec of pods.                           |
| code        | CodeSpec                          | body | TODO                                    |

### Code Example

**shell**

```sh
curl -X PATCH -H "Content-Type: application/json" -d '
{
  "meta": {
    "name": "tf-mnist-json",
    "namespace": "default",
    "framework": "TensorFlow",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
      "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "environment": {
    "image": "apache/submarine:tf-mnist-with-summaries-1.0"
  },
  "spec": {
    "Ps": {
      "replicas": 1,
      "resources": "cpu=1,memory=1024M"
    },
    "Worker": {
      "replicas": 2,
      "resources": "cpu=1,memory=2048M"
    }
  }
}
' http://127.0.0.1:32080/api/v1/experiment/experiment_1626160071451_0005
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment_1626160071451_0005",
    "name":"tf-mnist-json",
    "uid":"4944c603-0f21-49e5-826a-2ff820bb4d93",
    "status":"Accepted",
    "acceptedTime":"2021-07-13T16:57:27.000+08:00",
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
    "spec":{
      "meta":{
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{"ENV_1":"ENV1"}
      },
      "environment":{
        "name":null,
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":"apache/submarine:tf-mnist-with-summaries-1.0"
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"1024M","cpu":"1"}
        },
        "Worker":{
          "replicas":2,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"2048M","cpu":"1"}
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

### Delete Experiment

```
DELETE /api/v1/experiment/{id}
```

### Parameters

| Field Name | Type   | In   | Description    |
| ---------- | ------ | ---- | -------------- |
| id         | String | path | Experiment id. |

### Code Example

**shell**

```sh
curl -X DELETE http://127.0.0.1:32080/api/v1/experiment/experiment_1626160071451_0005
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment_1626160071451_0005",
    "name":"tf-mnist-json",
    "uid":"4944c603-0f21-49e5-826a-2ff820bb4d93",
    "status":"Deleted",
    "acceptedTime":null,
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
    "spec":{
      "meta":{
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{"ENV_1":"ENV1"}
      },
      "environment":{
        "name":null,
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":"apache/submarine:tf-mnist-with-summaries-1.0"
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"1024M","cpu":"1"}
        },
        "Worker":{
          "replicas":2,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{"memory":"2048M","cpu":"1"}
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

## List Experiment Log

```
GET /api/v1/experiment/logs
```

### Code Example

**shell**

```sh
curl -X GET http://127.0.0.1:32080/api/v1/experiment/logs
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":
  [{
    "experimentId":"experiment_1626160071451_0001",
    "logContent":
    [{
      "podName":"newexperiment1-ps-0",
      "podLog":[]
    },
    {
      "podName":"newexperiment1-worker-0",
      "podLog":[]
    }]
  }],
  "attributes":{}
}
```

## Get Experiment Log

```
GET /api/v1/experiment/logs/{id}
```

### Parameters

| Field Name | Type   | In   | Description    |
| ---------- | ------ | ---- | -------------- |
| id         | String | path | Experiment id. |

### Code Example

**shell**

```sh
curl -X GET http://127.0.0.1:32080/api/v1/experiment/logs/experiment_1626160071451_0001
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment_1626160071451_0001",
    "logContent":
    [{
      "podName":"newexperiment1-ps-0",
      "podLog":[]
    },
    {
      "podName":"newexperiment1-worker-0",
      "podLog":[]
    }]
  },
  "attributes":{}
}
```

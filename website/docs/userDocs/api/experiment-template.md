---
title: Experiment Template REST API
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

## Create Experiment Template

```
POST /api/v1/template
```

### Parameters

| Field Name     | Type                                | In   | Description                                 |
| -------------- | ----------------------------------- | ---- | ------------------------------------------- |
| name           | String                              | body | Experiment template name. This is required. |
| author         | String                              | body | Author name.                                |
| description    | String                              | body | Description of the experiment template.     |
| parameters     | List\<ExperimentTemplateParamSpec\> | body | Parameters of the experiment template.      |
| experimentSpec | ExperimentSpec                      | body | Spec of the experiment template.            |

#### **ExperimentTemplateParamSpec**

| Field Name  | Type    | Description                                      |
| ----------- | ------- | ------------------------------------------------ |
| name        | String  | Parameter name.                                  |
| required    | Boolean | true / false. Whether the parameter is required. |
| description | String  | Description of the parameter.                    |
| value       | String  | Value of the parameter.                          |

#### **ExperimentSpec**

| Field Name  | Type                              | Description                             |
| ----------- | --------------------------------- | --------------------------------------- |
| meta        | ExperimentMeta                    | Meta data of the experiment template.   |
| environment | EnvironmentSpec                   | Environment of the experiment template. |
| spec        | Map\<String, ExperimentTaskSpec\> | Spec of pods.                           |
| code        | CodeSpec                          | Experiment codespec.                    |

#### **ExperimentMeta**
| Field Name | Type                  | Description              |
| ---------- | --------------------- | ------------------------ |
| name       | String                | Experiment Name.         |
| namespace  | String                | Experiment namespace.    |
| framework  | String                | Experiment framework.    |
| cmd        | String                | Command.                 |
| envVars    | Map\<String, String\> | Environmental variables. |

#### **EnvironmentSpec**

See more details in [environment api](https://submarine.apache.org/docs/userDocs/api/environment).

#### **ExperimentTaskSpec**

| Field Name | Type                  | Description              |
| ---------- | --------------------- | ------------------------ |
| replicas   | Integer               | Numbers of replicas.     |
| resoureces | String                | Resouces of the task     |
| name       | String                | Task name.               |
| image      | String                | Image name.              |
| cmd        | String                | Command.                 |
| envVars    | Map\<String, String\> | Environmental variables. |

#### **CodeSpec**

| Field Name | Type   | Description             |
| ---------- | ------ | ----------------------- |
| syncMode   | String | sync mode of code spec. |
| url        | String | url of code spec.       |

### Code Example

**shell**

```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "name": "my-tf-mnist-template",
  "author": "author",
  "description": "This is a template to run tf-mnist",
  "parameters": [{
      "name": "learning_rate",
      "value": 0.1,
      "required": true,
      "description": "This is learning_rate of training."
    },
    {
      "name": "batch_size",
      "value": 150,
      "required": true,
      "description": "This is batch_size of training."
    },
    {
      "name": "experiment_name",
      "value": "tf-mnist1",
      "required": true,
      "description": "the name of experiment."
    }
  ],
  "experimentSpec": {
    "meta": {
      "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{learning_rate}} --batch_size={{batch_size}}",
      "name": "{{experiment_name}}",
      "envVars": {
        "ENV1": "ENV1"
      },
      "framework": "TensorFlow",
      "namespace": "default"
    },
    "spec": {
      "Ps": {
        "replicas": 1,
        "resources": "cpu=1,memory=1024M"
      },
      "Worker": {
        "replicas": 1,
        "resources": "cpu=1,memory=1024M"
      }
    },
    "environment": {
      "image": "apache/submarine:tf-mnist-with-summaries-1.0"
    }
  }
}
' http://127.0.0.1:32080/api/v1/template
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentTemplateId":{
      "id":1,
      "serverTimestamp":1626160071451
    },
    "experimentTemplateSpec":{
      "name":"my-tf-mnist-template",
      "author":"author",
      "description":"This is a template to run tf-mnist",
      "parameters":
      [{
          "name":"learning_rate",
          "required":"true",
          "description":"This is learning_rate of training.",
          "value":"0.1"
        },
        {
          "name":"batch_size",
          "required":"true",
          "description":"This is batch_size of training.",
          "value":"150"
        },
        {
          "name":"experiment_name",
          "required":"true",
          "description":"the name of experiment.",
          "value":"tf-mnist1"
        },
        {
          "name":"spec.Ps.replicas",
          "required":"false",
          "description":"",
          "value":"1"
        },
        {
          "name":"spec.Ps.resourceMap.cpu",
          "required":"false",
          "description":"",
          "value":"1"
        },
        {
          "name":"spec.Ps.resourceMap.memory",
          "required":"false",
          "description":"",
          "value":"1024M"
        },
        {
          "name":"spec.Worker.replicas",
          "required":"false",
          "description":"",
          "value":"1"
        },
        {
          "name":"spec.Worker.resourceMap.cpu",
          "required":"false",
          "description":"",
          "value":"1"
        },
        {
          "name":"spec.Worker.resourceMap.memory",
          "required":"false",
          "description":"","
          value":"1024M"
        }],
      "experimentSpec":{
        "meta":{
          "name":"{{experiment_name}}",
          "namespace":"default",
          "framework":"TensorFlow",
          "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d{{learning_rate}} --batch_size\u003d{{batch_size}}",
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
            "resourceMap":{
              "memory":"1024M",
              "cpu":"1"
            }
          },
          "Worker":{
            "replicas":1,
            "resources":"cpu\u003d1,memory\u003d1024M",
            "name":null,
            "image":null,
            "cmd":null,
            "envVars":null,
            "resourceMap":{
              "memory":"1024M",
              "cpu":"1"
            }
          }
        },
        "code":null
      }
    }
  },
  "attributes":{}
}
```

## List Experiment Template

```
GET /api/v1/template
```

### Code Example

**shell**

```sh
curl -X GET http://127.0.0.1:32080/api/v1/template
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    [{
      "experimentTemplateId":{
        "id":1,
        "serverTimestamp":1626160071451
      },
      "experimentTemplateSpec":{
        "name":"my-tf-mnist-template",
        "author":"author",
        "description":"This is a template to run tf-mnist",
        "parameters":
        [{
            "name":"learning_rate",
            "required":"true",
            "description":"This is learning_rate of training.",
            "value":"0.1"
          },
          {
            "name":"batch_size",
            "required":"true",
            "description":"This is batch_size of training.",
            "value":"150"
          },
          {
            "name":"experiment_name",
            "required":"true",
            "description":"the name of experiment.",
            "value":"tf-mnist1"
          },
          {
            "name":"spec.Ps.replicas",
            "required":"false",
            "description":"",
            "value":"1"
          },
          {
            "name":"spec.Ps.resourceMap.cpu",
            "required":"false",
            "description":"",
            "value":"1"
          },
          {
            "name":"spec.Ps.resourceMap.memory",
            "required":"false",
            "description":"",
            "value":"1024M"
          },
          {
            "name":"spec.Worker.replicas",
            "required":"false",
            "description":"",
            "value":"1"
          },
          {
            "name":"spec.Worker.resourceMap.cpu",
            "required":"false",
            "description":"",
            "value":"1"
          },
          {
            "name":"spec.Worker.resourceMap.memory",
            "required":"false",
            "description":"","
            value":"1024M"
          }],
        "experimentSpec":{
          "meta":{
            "name":"{{experiment_name}}",
            "namespace":"default",
            "framework":"TensorFlow",
            "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d{{learning_rate}} --batch_size\u003d{{batch_size}}",
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
              "resourceMap":{
                "memory":"1024M",
                "cpu":"1"
              }
            },
            "Worker":{
              "replicas":1,
              "resources":"cpu\u003d1,memory\u003d1024M",
              "name":null,
              "image":null,
              "cmd":null,
              "envVars":null,
              "resourceMap":{
                "memory":"1024M",
                "cpu":"1"
              }
            }
          },
          "code":null
        }
      }
    }],
  "attributes":{}
}
```

## Patch Experiment Template

```
PATCH /api/v1/template{name}
```

### Parameters

| Field Name     | Type                                | In            | Description                                 |
| -------------- | ----------------------------------- | ------------- | ------------------------------------------- |
| name           | String                              | path and body | Experiment template name. This is required. |
| author         | String                              | body          | Author name.                                |
| description    | String                              | body          | Description of the experiment template.     |
| parameters     | List\<ExperimentTemplateParamSpec\> | body          | Parameters of the experiment template.      |
| experimentSpec | ExperimentSpec                      | body          | Spec of the experiment template.            |

### Code Example

**shell**

```sh
curl -X PATCH -H "Content-Type: application/json" -d '
{
  "name": "my-tf-mnist-template",
  "author": "author-new",
  "description": "This is a template to run tf-mnist",
  "parameters": [{
      "name": "learning_rate",
      "value": 0.1,
      "required": true,
      "description": "This is learning_rate of training."
    },
    {
      "name": "batch_size",
      "value": 150,
      "required": true,
      "description": "This is batch_size of training."
    },
    {
      "name": "experiment_name",
      "value": "tf-mnist1",
      "required": true,
      "description": "the name of experiment."
    }
  ],
  "experimentSpec": {
    "meta": {
      "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{learning_rate}} --batch_size={{batch_size}}",
      "name": "{{experiment_name}}",
      "envVars": {
        "ENV1": "ENV1"
      },
      "framework": "TensorFlow",
      "namespace": "default"
    },
    "spec": {
      "Ps": {
        "replicas": 1,
        "resources": "cpu=1,memory=1024M"
      },
      "Worker": {
        "replicas": 1,
        "resources": "cpu=1,memory=1024M"
      }
    },
    "environment": {
      "image": "apache/submarine:tf-mnist-with-summaries-1.0"
    }
  }
}
' http://127.0.0.1:32080/api/v1/template/my-tf-mnist-template
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentTemplateId":{
      "id":2,
      "serverTimestamp":1626160071451
    },
    "experimentTemplateSpec":{
      "name":"my-tf-mnist-template",
      "author":"author-new",
      "description":"This is a template to run tf-mnist",
      "parameters":
      [{
        "name":"learning_rate",
        "required":"true",
        "description":"This is learning_rate of training.",
        "value":"0.1"
        },
        {
          "name":"batch_size",
          "required":"true",
          "description":"This is batch_size of training.",
          "value":"150"
        },
        {
          "name":"experiment_name",
          "required":"true",
          "description":"the name of experiment.",
          "value":"tf-mnist1"
        },
        {
          "name":"spec.Ps.replicas",
          "required":"false",
          "description":"",
          "value":"1"
        },
        {
          "name":"spec.Ps.resourceMap.cpu",
          "required":"false",
          "description":"",
          "value":"1"
        },
        {
          "name":"spec.Ps.resourceMap.memory",
          "required":"false",
          "description":"",
          "value":"1024M"
        },
        {
          "name":"spec.Worker.replicas",
          "required":"false",
          "description":"",
          "value":"1"
        },
        {
          "name":"spec.Worker.resourceMap.cpu",
          "required":"false",
          "description":"",
          "value":"1"
        },
        {
          "name":"spec.Worker.resourceMap.memory",
          "required":"false",
          "description":"",
          "value":"1024M"
      }],
      "experimentSpec":{
        "meta":{
          "name":"{{experiment_name}}",
          "namespace":"default",
          "framework":"TensorFlow",
          "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d{{learning_rate}} --batch_size\u003d{{batch_size}}",
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
    }
  },
  "attributes":{}
}
```

## Delete Experiment Template

```
DELETE /api/v1/template{name}
```

### Parameters
| Field Name | Type   | In   | Description                                 |
| ---------- | ------ | ---- | ------------------------------------------- |
| name       | String | path | Experiment template name. This is required. |

### Code Example

**shell**

```sh
curl -X DELETE http://127.0.0.1:32080/api/v1/template/my-tf-mnist-template
```

**reponse**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentTemplateId":{
      "id":2,
      "serverTimestamp":1626160071451
    },
    "experimentTemplateSpec":{
      "name":"my-tf-mnist-template",
      "author":"author-new",
      "description":"This is a template to run tf-mnist",
      "parameters":
      [{
        "name":"learning_rate",
        "required":"true",
        "description":"This is learning_rate of training.",
        "value":"0.1"
      },
      {
        "name":"batch_size",
        "required":"true",
        "description":"This is batch_size of training.",
        "value":"150"
      },
      {
        "name":"experiment_name",
        "required":"true",
        "description":"the name of experiment.",
        "value":"tf-mnist1"
      },
      {
        "name":"spec.Ps.replicas",
        "required":"false",
        "description":"",
        "value":"1"
      },
      {
        "name":"spec.Ps.resourceMap.cpu",
        "required":"false",
        "description":"",
        "value":"1"
      },
      {
        "name":"spec.Ps.resourceMap.memory",
        "required":"false",
        "description":"",
        "value":"1024M"
      },
      {
        "name":"spec.Worker.replicas",
        "required":"false",
        "description":"",
        "value":"1"
      },
      {
        "name":"spec.Worker.resourceMap.cpu",
        "required":"false",
        "description":"",
        "value":"1"
      },
      {
        "name":"spec.Worker.resourceMap.memory",
        "required":"false",
        "description":"",
        "value":"1024M"
      }],
      "experimentSpec":{
        "meta":{
          "name":"{{experiment_name}}",
          "namespace":"default",
          "framework":"TensorFlow",
          "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d{{learning_rate}} --batch_size\u003d{{batch_size}}",
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
    }
  },
  "attributes":{}
}
```

## Use Template to Create a Experiment

```
POST /api/v1/experiment/{template_name}
```

### Parameters

| Field Name    | Type                | In   | Description                                               |
| ------------- | ------------------- | ---- | --------------------------------------------------------- |
| template_name | String              | path | Experiment template name.                                 |
| name          | String              | body | Experiment template name.                                 |
| params        | Map<String, String> | body | Parameters of the experiment including `experiment_name`. |

### Code Example

**shell**

```sh
curl -X POST -H "Content-Type: application/json" -d '
{
    "name": "tf-mnist",
    "params": {
        "learning_rate":"0.01",
        "batch_size":"150",
        "experiment_name":"newexperiment1"
    }
}
' http://127.0.0.1:32080/api/v1/experiment/my-tf-mnist-template
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
    "name":"newexperiment1",
    "uid":"b895985c-411c-4e89-90e0-c60a2a8a4235",
    "status":"Accepted",
    "acceptedTime":"2021-07-13T16:21:31.000+08:00",
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
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
  "attributes":{}
}
```

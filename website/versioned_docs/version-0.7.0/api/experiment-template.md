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

:::caution
Note: The Experiment API is in the alpha stage which is subjected to incompatible changes in future releases.
:::

Developers can register a parameterized experiment as an experiment template,
For example, if the developer wants to change the following "--learning_rate=0.1" to parameters.
```json
"experimentSpec": {
  "meta": {
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.1 --batch_size=150"
  }, 
  "...": "..."
}
```

They can use two curly braces as placeholders, the template format will be as
```json
"experimentSpec": {
  "meta": {
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{learning_rate}} --batch_size=150"
  }, 
  "...": "..."
}
```

The template parameters format will be as
```json
{
  "name": "learning_rate",
  "value": 0.1,
  "required": true,
  "description": "This is learning_rate of training."
}
```
name: placeholder name
value; default value
required: Indicates whether the user must enter parameters, when required is true, value can be null
description: Introduction of this parameter

Users can use existing experiment templates and adjust the default value to create experiments.
After the user submits the experiment template, the submarine server finds the corresponding template based on the name. And the template handler converts input parameters to an actual experiment, such as a distributed TF experiment.

The "replicas", "cpu", "memory" of resources will be automatically parameterized, so developers do not need to add them.
For example, if there are "Ps" and "Worker" under spec, the following parameters will be automatically appended.
```
spec.Ps.replicas
spec.Ps.resourceMap.cpu
spec.Ps.resourceMap.memory
spec.Worker.replicas
spec.Worker.resourceMap.cpu
spec.Worker.resourceMap.memory
```


## Create experiment template
`POST /api/v1/template`

### Parameters
| Field Name     | Type                               | In   | Description                                 |
| -------------- | ---------------------------------- | ---- | ------------------------------------------- |
| name           | String                             | body | Experiment template name. This is required. |
| author         | String                             | body | Author name.                                |
| description    | String                             | body | Description of the experiment template.     |
| parameters     | List<ExperimentTemplateParamSpec\> | body | Parameters of the experiment template.      |
| experimentSpec | ExperimentSpec                     | body | Spec of the experiment template.            |

#### **ExperimentTemplateParamSpec**

| Field Name  | Type    | Description                                      |
| ----------- | ------- | ------------------------------------------------ |
| name        | String  | Parameter name.                                  |
| required    | Boolean | true / false. Whether the parameter is required. |
| description | String  | Description of the parameter.                    |
| value       | String  | Value of the parameter.                          |

#### **ExperimentSpec**

| Field Name  | Type                             | Description                             |
| ----------- | -------------------------------- | --------------------------------------- |
| meta        | ExperimentMeta                   | Meta data of the experiment template.   |
| environment | EnvironmentSpec                  | Environment of the experiment template. |
| spec        | Map<String, ExperimentTaskSpec\> | Spec of pods.                           |
| code        | CodeSpec                         | Experiment codespec.                    |

#### **ExperimentMeta**
| Field Name | Type                 | Description              |
| ---------- | -------------------- | ------------------------ |
| name       | String               | Experiment Name.         |
| namespace  | String               | Experiment namespace.    |
| framework  | String               | Experiment framework.    |
| cmd        | String               | Command.                 |
| envVars    | Map<String, String\> | Environmental variables. |

#### **EnvironmentSpec**

See more details in [environment api](environment.md).

#### **ExperimentTaskSpec**

| Field Name | Type                 | Description              |
| ---------- | -------------------- | ------------------------ |
| replicas   | Integer              | Numbers of replicas.     |
| resoureces | String               | Resouces of the task     |
| name       | String               | Task name.               |
| image      | String               | Image name.              |
| cmd        | String               | Command.                 |
| envVars    | Map<String, String\> | Environmental variables. |

#### **CodeSpec**

| Field Name | Type   | Description             |
| ---------- | ------ | ----------------------- |
| syncMode   | String | sync mode of code spec. |
| url        | String | url of code spec.       |
### Example

<details>
<summary>Example Request</summary>
<div>

```shell
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
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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

</div>
</details>

## List experiment template
`GET /api/v1/template`

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X GET http://127.0.0.1:32080/api/v1/template
```
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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

</div>
</details>


## Get experiment template
`GET /api/v1/template/{name}`

### Parameters

| Field Name | Type   | In   | Description               | Required |
| ---------- | ------ | ---- | ------------------------- | :------: |
| name       | String | path | Experiment template name. |    o     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X GET http://127.0.0.1:32080/api/v1/template/my-tf-mnist-template
```
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentTemplateId":{
      "id":1,
      "serverTimestamp":1650788898882
    },
    "experimentTemplateSpec":{
      "name":"my-tf-mnist-template",
      "author":"author",
      "description":"This is a template to run tf-mnist",
      "parameters":[
        {
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
        }
      ],
      "experimentSpec":{
        "meta":{
          "experimentId":null,
          "name":"{{experiment_name}}",
          "namespace":"default",
          "framework":"TensorFlow",
          "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d{{learning_rate}} --batch_size\u003d{{batch_size}}",
          "envVars":{
            "ENV1":"ENV1"
          },
          "tags":[]
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

</div>
</details>

## Patch template
`PATCH /api/v1/template/{name}`

### Parameters

| Field Name     | Type                               | In            | Description                             | Required |
| -------------- | ---------------------------------- | ------------- | --------------------------------------- | :------: |
| name           | String                             | path and body | Experiment template name.               |    o     |
| author         | String                             | body          | Author name.                            |    o     |
| description    | String                             | body          | Description of the experiment template. |    x     |
| parameters     | List<ExperimentTemplateParamSpec\> | body          | Parameters of the experiment template.  |    o     |
| experimentSpec | ExperimentSpec                     | body          | Spec of the experiment template.        |    o     |

### Example
<details>
<summary>Example Request</summary>
<div>

```shell
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
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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

</div>
</details>

:::note
"description", "parameters", "experimentSpec", "author" etc can be updated using this API.
"name" of experiment template is not supported.
:::


## Delete template
`GET /api/v1/template/{name}`

### Parameters

| Field Name | Type   | In   | Description               | Required |
| ---------- | ------ | ---- | ------------------------- | :------: |
| name       | String | path | Experiment template name. |    o     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X DELETE http://127.0.0.1:32080/api/v1/template/my-tf-mnist-template
```
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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

</div>
</details>

## Use template to create a experiment
`POST /api/v1/experiment/{template_name}`

### Parameters

| Field Name | Type                | In            | Description                                               | Required |
| ---------- | ------------------- | ------------- | --------------------------------------------------------- | :------: |
| name       | String              | path and body | Experiment template name.                                 |    o     |
| params     | Map<String, String> | body          | Parameters of the experiment including `experiment_name`. |    o     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
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
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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


</div>
</details>

---
title: Notebook REST API
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
The Notebook API is in the alpha stage which is subjected to incompatible changes in future releases.
:::

## Create a notebook instance
`POST /api/v1/notebook`

### Parameters

NotebookSpec in request body.

#### **NotebookSpec**

| Field Name  | Type            | Description                             | Required |
| ----------- | --------------- | --------------------------------------- | :------: |
| meta        | NotebookMeta    | Meta data of the notebook.              |    o     |
| environment | EnvironmentSpec | Environment of the experiment template. |    o     |
| spec        | NotebookPodSpec | Spec of the notebook pods.              |    o     |

#### **NotebookMeta**

| Field Name | Type   | Description         | Required |
| ---------- | ------ | ------------------- | :------: |
| name       | String | Notebook name.      |    o     |
| namespace  | String | Notebook namespace. |    o     |
| ownerId    | String | User id.            |    o     |

#### **EnvironmentSpec**

See more details in [environment api](environment.md).

#### **NotebookPodSpec**

| Field Name | Type                 | Description              | Required |
| ---------- | -------------------- | ------------------------ | :------: |
| envVars    | Map<String, String\> | Environmental variables. |    x     |
| resources  | String               | Resourecs of the pod.    |    o     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X POST -H "Content-Type: application/json" -d '
{
  "meta": {
    "name": "test-nb",
    "namespace": "default",
    "ownerId": "e9ca23d68d884d4ebb19d07889727dae"
  },
  "environment": {
    "name": "notebook-env"
  },
  "spec": {
    "envVars": {
      "TEST_ENV": "test"
    },
    "resources": "cpu=1,memory=1.0Gi"
  }
}
' http://127.0.0.1:32080/api/v1/notebook
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
  "message":"Create a notebook instance",
  "result":{
    "notebookId":"notebook_1647574374688_0001",
    "name":"test-nb",
    "uid":"4a839fef-b4c9-483a-b4e8-c17236588118",
    "url":"/notebook/default/test-nb/lab",
    "status":"creating",
    "reason":"The notebook instance is creating",
    "createdTime":"2022-03-18T16:13:16.000+08:00",
    "deletedTime":null,
    "spec":{
      "meta":{
        "name":"test-nb",
        "namespace":"default",
        "ownerId":"e9ca23d68d884d4ebb19d07889727dae",
        "labels":{
          "notebook-owner-id":"e9ca23d68d884d4ebb19d07889727dae",
          "notebook-id":"notebook_1647574374688_0001"
        }
      },
      "environment":{
        "name":"notebook-env",
        "dockerImage":"apache/submarine:jupyter-notebook-0.7.0",
        "kernelSpec":{
          "name":"submarine_jupyter_py3",
          "channels":[
            "defaults"
          ],
          "condaDependencies":[],
          "pipDependencies":[]
        },
        "description":null,
        "image":null
      },
      "spec":{
        "envVars":{
          "TEST_ENV":"test"
        },
        "resources":"cpu\u003d1,memory\u003d1.0Gi"
      }
    }
  },
  "attributes":{}
}
```
</div>
</details>


## List notebook instances which belong to user
`GET /api/v1/notebook?id={user_id}`

### Parameters

| Field Name | Type   | In    | Description | Required |
| ---------- | ------ | ----- | ----------- | :------: |
| id         | String | query | User id.    |    o     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X GET http://127.0.0.1:32080/api/v1/notebook?id={user_id}
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
  "message":"List all notebook instances",
  "result":[
    {
      "notebookId":"notebook_1647574374688_0001",
      "name":"test-nb",
      "uid":null,
      "url":"/notebook/default/test-nb/lab",
      "status":"running",
      "reason":"The notebook instance is running",
      "createdTime":"2022-03-18T16:13:16.000+08:00",
      "deletedTime":"2022-03-18T16:13:21.000+08:00",
      "spec":{
        "meta":{
          "name":"test-nb",
          "namespace":"default",
          "ownerId":"e9ca23d68d884d4ebb19d07889727dae",
          "labels":{
            "notebook-owner-id":"e9ca23d68d884d4ebb19d07889727dae",
            "notebook-id":"notebook_1647574374688_0001"
          }
        },
        "environment":{
          "name":"notebook-env",
          "dockerImage":"apache/submarine:jupyter-notebook-0.7.0",
          "kernelSpec":{
            "name":"submarine_jupyter_py3",
            "channels":[
              "defaults"
            ],
            "condaDependencies":[],
            "pipDependencies":[]
          },
          "description":null,
          "image":null
        },
        "spec":{
          "envVars":{
            "TEST_ENV":"test"
          },
          "resources":"cpu\u003d1,memory\u003d1.0Gi"
        }
      }
    }
  ],
  "attributes":{}
}
```
</div>
</details>

## Get the notebook instance
`GET /api/v1/notebook/{id}`

### Parameters

| Field Name | Type   | In   | Description  | Required |
| ---------- | ------ | ---- | ------------ | :------: |
| id         | String | path | Notebook id. |    o     |
### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X GET http://127.0.0.1:32080/api/v1/notebook/{id}
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
  "message":"Get the notebook instance",
  "result":{
    "notebookId":"notebook_1647574374688_0001",
    "name":"test-nb",
    "uid":"4a839fef-b4c9-483a-b4e8-c17236588118",
    "url":"/notebook/default/test-nb/lab",
    "status":"running",
    "reason":"The notebook instance is running",
    "createdTime":"2022-03-18T16:13:16.000+08:00",
    "deletedTime":"2022-03-18T16:13:21.000+08:00",
    "spec":{
      "meta":{
        "name":"test-nb",
        "namespace":"default",
        "ownerId":"e9ca23d68d884d4ebb19d07889727dae",
        "labels":{
          "notebook-owner-id":"e9ca23d68d884d4ebb19d07889727dae",
          "notebook-id":"notebook_1647574374688_0001"
        }
      },
      "environment":{
        "name":"notebook-env",
        "dockerImage":"apache/submarine:jupyter-notebook-0.7.0",
        "kernelSpec":{
          "name":"submarine_jupyter_py3",
          "channels":[
            "defaults"
          ],
          "condaDependencies":[],
          "pipDependencies":[]
        },
        "description":null,
        "image":null
      },
      "spec":{
        "envVars":{
          "TEST_ENV":"test"
        },
        "resources":"cpu\u003d1,memory\u003d1.0Gi"
      }
    }
  },
  "attributes":{}
}
```
</div>
</details>

## Delete the notebook instance
`DELETE /api/v1/notebook/{id}`

### Parameters

| Field Name | Type   | In   | Description  | Required |
| ---------- | ------ | ---- | ------------ | :------: |
| id         | String | path | Notebook id. |    o     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X DELETE http://127.0.0.1:32080/api/v1/notebook/{id}
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
  "message":"Delete the notebook instance",
  "result":{
    "notebookId":"notebook_1647574374688_0001",
    "name":"test-nb",
    "uid":"4a839fef-b4c9-483a-b4e8-c17236588118",
    "url":"/notebook/default/test-nb/lab",
    "status":"terminating",
    "reason":"The notebook instance is terminating",
    "createdTime":"2022-03-18T16:13:16.000+08:00",
    "deletedTime":"2022-03-18T16:13:21.000+08:00",
    "spec":{
      "meta":{
        "name":"test-nb",
        "namespace":"default",
        "ownerId":"e9ca23d68d884d4ebb19d07889727dae",
        "labels":{
          "notebook-owner-id":"e9ca23d68d884d4ebb19d07889727dae",
          "notebook-id":"notebook_1647574374688_0001"
        }
      },
      "environment":{
        "name":"notebook-env",
        "dockerImage":"apache/submarine:jupyter-notebook-0.7.0",
        "kernelSpec":{
          "name":"submarine_jupyter_py3",
          "channels":[
            "defaults"
          ],
          "condaDependencies":[],
          "pipDependencies":[]
        },
        "description":null,
        "image":null
      },
      "spec":{
        "envVars":{
          "TEST_ENV":"test"
        },
        "resources":"cpu\u003d1,memory\u003d1.0Gi"
      }
    }
  },
  "attributes":{}
}
```
</div>
</details>

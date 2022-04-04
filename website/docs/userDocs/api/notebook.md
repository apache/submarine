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

## Create a Notebook Instance

```
POST /api/v1/notebook
```

### Parameters

NotebookSpec in request body.

#### **NotebookSpec**

| Field Name  | Type            | Description                             |
| ----------- | --------------- | --------------------------------------- |
| meta        | NotebookMeta    | Meta data of the notebook.              |
| environment | EnvironmentSpec | Environment of the experiment template. |
| spec        | NotebookPodSpec | Spec of the notebook pods.              |

#### **NotebookMeta**

| Field Name | Type   | Description         |
| ---------- | ------ | ------------------- |
| name       | String | Notebook name.      |
| namespace  | String | Notebook namespace. |
| ownerId    | String | User id.            |

#### **EnvironmentSpec**

See more details in [environment api](environment.md).

#### **NotebookPodSpec**

| Field Name | Type                  | Description              |
| ---------- | --------------------- | ------------------------ |
| envVars    | Map<String, String\>  | Environmental variables. |
| resources  | String                | Resourecs of the pod.    |

### Code Example

**shell**

```sh
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

**response:**

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
        "dockerImage":"apache/submarine:jupyter-notebook-0.8.0-SNAPSHOT",
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

## List notebook instances which belong to user

```
GET /api/v1/notebook
```

### Parameters

| Field Name | Type   | In    | Description |
| ---------- | ------ | ----- | ----------- |
| id         | String | query | User id.    |

### Code Example

**shell**

```sh
curl -X GET http://127.0.0.1:32080/api/v1/notebook?id={user_id}
```

**response**

```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":"List all notebook instances",
  "result":
  [{
    "notebookId":"notebook_1626160071451_0001",
    "name":"test-nb",
    "uid":"a56713da-f2a3-40d0-ae2e-45fdc0bb15f5",
    "url":"/notebook/default/test-nb/lab",
    "status":"waiting",
    "reason":"ContainerCreating",
    "createdTime":"2021-07-13T16:23:38.000+08:00",
    "deletedTime":null,
    "spec":{
      "meta":{
        "name":"test-nb",
        "namespace":"default",
        "ownerId":"e9ca23d68d884d4ebb19d07889727dae"
      },
      "environment":{
        "name":"notebook-env",
        "dockerImage":"apache/submarine:jupyter-notebook-0.8.0-SNAPSHOT",
        "kernelSpec":{
          "name":"submarine_jupyter_py3",
          "channels":["defaults"],
          "condaDependencies":[],
          "pipDependencies":[]
        },
        "description":null,
        "image":null
      },
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

## Get the notebook instance

```
GET /api/v1/notebook/{id}
```

### Parameters

| Field Name | Type   | In   | Description  |
| ---------- | ------ | ---- | ------------ |
| id         | String | path | Notebook id. |

### Code Example

**shell**

```sh
curl -X GET http://127.0.0.1:32080/api/v1/notebook/notebook_1626160071451_0001
```

**response**

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
        "dockerImage":"apache/submarine:jupyter-notebook-0.8.0-SNAPSHOT",
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

## Delete the notebook instance

```
DELETE /api/v1/notebook/{id}
```

### Parameters

| Field Name | Type   | In   | Description  |
| ---------- | ------ | ---- | ------------ |
| id         | String | path | Notebook id. |

### Code Example

**shell**

```sh
curl -X DELETE http://127.0.0.1:32080/api/v1/notebook/notebook_1626160071451_0001
```

**response:**

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
        "dockerImage":"apache/submarine:jupyter-notebook-0.8.0-SNAPSHOT",
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

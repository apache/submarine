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

# Notebook API Reference

> Note: The Notebook API is in the alpha stage which is subjected to incompatible changes in future releases.

## Create a notebook instance
`POST /api/v1/notebook`

**Example Request**
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

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":"Create a notebook instance",
  "result":{
    "notebookId":"notebook_1597931805405_0001",
    "name":"test-nb",
    "uid":"5a94c01d-6a92-4222-bc66-c610c277546d",
    "url":"/notebook/default/test-nb/",
    "status": null,
    "reason": null,
    "createdTime":"2020-08-20T21:58:27.000+08:00",
    "deletedTime":null,
    "spec":{
      "meta":{
        "name":"test-nb",
        "namespace":"default",
        "ownerId":"e9ca23d68d884d4ebb19d07889727dae"
      },
      "environment":{
        "name":"notebook-env",
        "dockerImage":"apache/submarine:jupyter-notebook-0.5.0",
        "kernelSpec":{
          "name": "team_default_python_3.7",
          "channels": [
            "defaults"
          ],
          "dependencies": [
            ""
          ]
        },
        "description":null,
        "image":null
      },
      "spec":{
        "envVars":{
          "TEST_ENV":"test"
        },
        "resources":"cpu=1,memory=1.0Gi"
      }
    }
  },
  "attributes":{}
}
```

## List notebook instances which belong to user
`GET /api/v1/notebook`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/notebook?id={user_id}
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":"List all notebook instances",
  "result":[
    {
      "notebookId":"notebook_1597931805405_0001",
      "name":"test-nb",
      "uid":"5a94c01d-6a92-4222-bc66-c610c277546d",
      "url":"/notebook/default/test-nb/",
      "status": "ready",
      "reason": "Notebook instance is running",
      "createdTime":"2020-08-20T21:58:27.000+08:00",
      "deletedTime":null,
      "spec":{
        "meta":{
          "name":"test-nb",
          "namespace":"default",
          "ownerId":"e9ca23d68d884d4ebb19d07889727dae"
        },
        "environment":{
          "name":"notebook-env",
          "dockerImage":"apache/submarine:jupyter-notebook-0.5.0",
          "kernelSpec":{
            "name": "team_default_python_3.7",
            "channels": [
              "defaults"
            ],
            "dependencies": [
              ""
            ]
          },
          "description":null,
          "image":null
        },
        "spec":{
          "envVars":{
            "TEST_ENV":"test"
          },
          "resources":"cpu=1,memory=1.0Gi"
        }
      }
    }
  ],
  "attributes":{}
}
```

## Get the notebook instance
`GET /api/v1/notebook/{id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/notebook/{id}
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":"Get the notebook instance",
  "result":{
    "notebookId":"notebook_1597931805405_0001",
    "name":"test-nb",
    "uid":"5a94c01d-6a92-4222-bc66-c610c277546d",
    "url":"/notebook/default/test-nb/",
    "status":"Created",
    "createdTime":"2020-08-20T21:58:27.000+08:00",
    "deletedTime":null,
    "spec":{
      "meta":{
        "name":"test-nb",
        "namespace":"default",
        "ownerId":"e9ca23d68d884d4ebb19d07889727dae"
      },
      "environment":{
        "name":"notebook-env",
        "dockerImage":"apache/submarine:jupyter-notebook-0.5.0",
        "kernelSpec":{
          "name": "team_default_python_3.7",
          "channels": [
            "defaults"
          ],
          "dependencies": [
            ""
          ]
        },
        "description":null,
        "image":null
      },
      "spec":{
        "envVars":{
          "TEST_ENV":"test"
        },
        "resources":"cpu=1,memory=1.0Gi"
      }
    }
  },
  "attributes":{}
}
```

## Delete the notebook instance
`DELETE /api/v1/notebook/{id}`

**Example Request:**
```sh
curl -X DELETE http://127.0.0.1:8080/api/v1/notebook/{id}
```

**Example Response:**
```json
{
  "status": "OK",
  "code": 200,
  "success": true,
  "message": "Delete the notebook instance",
  "result": {
    "notebookId": "notebook_1597931805405_0001",
    "name": "test-nb",
    "uid": "5a94c01d-6a92-4222-bc66-c610c277546d",
    "url": "/notebook/default/test-nb/",
    "status": "Deleted",
    "createdTime": "2020-08-22T14:03:19.000+08:00",
    "deletedTime": "2020-08-22T14:46:28+0800",
    "spec": {
      "meta": {
        "name": "test-nb",
        "namespace": "default",
        "ownerId":"e9ca23d68d884d4ebb19d07889727dae"
      },
      "environment": {
        "name": "notebook-env",
        "dockerImage": "apache/submarine:jupyter-notebook-0.5.0",
        "kernelSpec": {
          "name": "team_default_python_3.7",
          "channels": [
            "defaults"
          ],
          "dependencies": [
            ""
          ]
        },
        "description": null,
        "image": null
      },
      "spec": {
        "envVars": {
          "TEST_ENV": "test"
        },
        "resources": "cpu=1,memory=1.0Gi"
      }
    }
  },
  "attributes": {}
}
```

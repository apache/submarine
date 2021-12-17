---
title: Serve REST API
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

> Note: The Model Version API is in the alpha stage which is subjected to incompatible changes in future releases.

### List model versions under a registered model
`GET /api/v1/model-version/{name}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/model-version/register
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "List all model version instances",
   "result" : [
      {
         "creationTime" : "2021-12-12 02:27:05",
         "currentStage" : "None",
         "dataset" : null,
         "description" : null,
         "experimentId" : "experiment-1639276018590-0001",
         "lastUpdatedTime" : "2021-12-12 02:27:05",
         "modelType" : "tensorflow",
         "name" : "register",
         "source" : "s3://submarine/experiment-1639276018590-0001/example/1",
         "tags" : [],
         "userId" : "",
         "version" : 1
      },
      {
         "creationTime" : "2021-12-12 02:27:05",
         "currentStage" : "None",
         "dataset" : null,
         "description" : null,
         "experimentId" : "experiment-1639276018590-0001",
         "lastUpdatedTime" : "2021-12-12 02:27:05",
         "modelType" : "tensorflow",
         "name" : "register",
         "source" : "s3://submarine/experiment-1639276018590-0001/example/2",
         "tags" : [],
         "userId" : "",
         "version" : 2
      },
      {
         "creationTime" : "2021-12-12 02:27:05",
         "currentStage" : "None",
         "dataset" : null,
         "description" : null,
         "experimentId" : "experiment-1639276018590-0001",
         "lastUpdatedTime" : "2021-12-12 02:27:05",
         "modelType" : "tensorflow",
         "name" : "register",
         "source" : "s3://submarine/experiment-1639276018590-0001/example1/1",
         "tags" : [],
         "userId" : "",
         "version" : 3
      },
      {
         "creationTime" : "2021-12-12 02:27:06",
         "currentStage" : "None",
         "dataset" : null,
         "description" : null,
         "experimentId" : "experiment-1639276018590-0001",
         "lastUpdatedTime" : "2021-12-12 02:27:06",
         "modelType" : "tensorflow",
         "name" : "register",
         "source" : "s3://submarine/experiment-1639276018590-0001/example2/1",
         "tags" : [],
         "userId" : "",
         "version" : 4
      },
   ],
   "status" : "OK",
   "success" : true
}
```

### Get a model version
`GET /api/v1/model-version/{name}/{version}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/model-version/register/1
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "Get the model version instance",
   "result" : {
      "creationTime" : "2021-12-12 02:27:05",
      "currentStage" : "None",
      "dataset" : null,
      "description" : null,
      "experimentId" : "experiment-1639276018590-0001",
      "lastUpdatedTime" : "2021-12-12 02:27:05",
      "modelType" : "tensorflow",
      "name" : "register",
      "source" : "s3://submarine/experiment-1639276018590-0001/example/1",
      "tags" : [],
      "userId" : "",
      "version" : 1
   },
   "status" : "OK",
   "success" : true
}
```

### Patch a model version
`PATCH /api/v1/model-version`

**Example Request:**
```sh
curl -X PATCH -H "Content-Type: application/json" -d '
{
    "name": "register",
    "version": 1,
    "description": "new_description",
    "currentStage": "production",
    "dataset": "new_dataset"
}' http://127.0.0.1:32080/api/v1/model-version
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "Update the model version instance",
   "result" : null,
   "status" : "OK",
   "success" : true
}
```

## Delete a model version
`DELETE /api/v1/model-version/{name}/{version}`

**Example Request**
```sh
curl -X DELETE http://127.0.0.1:32080/api/v1/model-version/register/1
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "Delete the model version instance",
   "result" : null,
   "status" : "OK",
   "success" : true
}
```

## Create a model version tag
`POST /api/v1/model-version/tag?name={name}&version={version}&tag={tag}`

**Example Request**
```sh
curl -X POST http://127.0.0.1:32080/api/v1/model-version/tag?name=register&version=2&tag=789
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "Create a model version tag instance",
   "result" : null,
   "status" : "OK",
   "success" : true
}
```

## Delete a model version tag
`DELETE /api/v1/model-version/tag?name={name}&version={version}&tag={tag}`

**Example Request**
```sh
curl -X DELETE http://127.0.0.1:32080/api/v1/model-version/tag?name=register&version=2&tag=789
```

**Example Response:**
```json
{
    "status":"OK",
    "code":200,
    "success":true,
    "message":"Delete a registered model tag instance",
    "result":null,
    "attributes":{}
}
```
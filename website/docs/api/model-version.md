---
title: Model Version REST API
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
The Model Version API is in the alpha stage which is subjected to incompatible changes in future releases.
:::

## Create a model version
`POST /api/v1/model-version?baseDir={baseDir}`

### Parameters

| Field Name   | Type          | In   | Description                               | Required |
| ------------ | ------------- | ---- | ----------------------------------------- | :------: |
| baseDir      | String        | path | experiment directory path.                |    o     |
| name         | String        | body | registered model name.                    |    o     |
| experimentId | String        | body | Add a tag for the registered model.       |    o     |
| description  | String        | body | Add description for the version of model. |    x     |
| tags         | List<String\> | body | Add tags for the version of model.        |    x     |
### Example

## List model versions under a registered model
`GET /api/v1/model-version/{name}`

### Parameters

| Field Name | Type   | In   | Description            | Required |
| ---------- | ------ | ---- | ---------------------- | :------: |
| name       | String | path | registered model name. |    o     |

### Example
<details>
<summary>Example Request</summary>
<div>

```shell
curl -X GET http://127.0.0.1:32080/api/v1/model-version/register
```
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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
   ],
   "status" : "OK",
   "success" : true
}
```
</div>
</details>

## Get a model version
`GET /api/v1/model-version/{name}/{version}`

### Parameters

| Field Name | Type   | In   | Description               | Required |
| ---------- | ------ | ---- | ------------------------- | :------: |
| name       | String | path | Registered model name.    |    o     |
| version    | String | path | Registered model version. |    o     |
### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X GET http://127.0.0.1:32080/api/v1/model-version/register/1
```
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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
</div>
</details>

## Patch a model version
`PATCH /api/v1/model-version`

### Parameters

| Field Name   | Type   | In   | Description               | Required |
| ------------ | ------ | ---- | ------------------------- | :------: |
| name         | String | body | Registered model name.    |    o     |
| version      | String | body | Registered model version. |    o     |
| description  | String | body | New description.          |    x     |
| currentStage | String | body | Stage of the model.       |    x     |
| dataset      | String | body | Dataset use in the model. |    x     |

### Example
<details>
<summary>Example Request</summary>
<div>

```shell
curl -X PATCH -H "Content-Type: application/json" -d '
{
    "name": "register",
    "version": 1,
    "description": "new_description",
    "currentStage": "production",
    "dataset": "new_dataset"
}' http://127.0.0.1:32080/api/v1/model-version
```
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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
</div>
</details>

## Delete a model version
`DELETE /api/v1/model-version/{name}/{version}`

### Parameters

| Field Name | Type   | In   | Description               | Required |
| ---------- | ------ | ---- | ------------------------- | :------: |
| name       | String | path | Registered model name.    |    o     |
| version    | String | path | Registered model version. |    o     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X DELETE http://127.0.0.1:32080/api/v1/model-version/register/1
```
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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
</div>
</details>

## Create a model version tag
`POST /api/v1/model-version/tag?name={name}&version={version}&tag={tag}`

### Parameters

| Field Name | Type   | In    | Description                          | Required |
| ---------- | ------ | ----- | ------------------------------------ | :------: |
| name       | String | query | Registered model name.               |    o     |
| version    | String | query | Registered model version.            |    o     |
| tag        | String | query | Tag of the registered model version. |    o     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X POST http://127.0.0.1:32080/api/v1/model-version/tag?name=register&version=2&tag=789
```
</div>
</details>

<details>
<summary>Example Response</summary>
<div>

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
</div>
</details>

## Delete a model version tag
`DELETE /api/v1/model-version/tag?name={name}&version={version}&tag={tag}`

### Parameters

| Field Name | Type   | In    | Description                          | Required |
| ---------- | ------ | ----- | ------------------------------------ | :------: |
| name       | String | query | Registered model name.               |    o     |
| version    | String | query | Registered model version.            |    o     |
| tag        | String | query | Tag of the registered model version. |    o     |

### Example
<details>
<summary>Example Request</summary>
<div>

```shell
curl -X DELETE http://127.0.0.1:32080/api/v1/model-version/tag?name=register&version=2&tag=789
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
    "message":"Delete a registered model tag instance",
    "result":null,
    "attributes":{}
}
```
</div>
</details>

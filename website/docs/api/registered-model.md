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

> Note: The Registered Model API is in the alpha stage which is subjected to incompatible changes in future releases.

## Create a registered model
`POST /api/v1/registered-model`

**Example Request**
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
   "name": "example_name",
   "description": "example_description",
   "tags": ["123", "456"]
   }
' http://127.0.0.1:32080/api/v1/registered-model
```

**Example Response:**
```json
{
    "status":"OK",
    "code":200,
    "success":true,
    "message":"Create a registered model instance",
    "result":null,
    "attributes":{}
}
```

### List registered models
`GET /api/v1/registered-model`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/registered-model
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "List all registered model instances",
   "result" : [
      {
         "creationTime" : "2021-12-16 10:14:06",
         "description" : "example_description",
         "lastUpdatedTime" : "2021-12-16 10:14:06",
         "name" : "example_name",
         "tags" : [
            "123",
            "456"
         ]
      },
      {
         "creationTime" : "2021-12-16 10:16:25",
         "description" : "example_description",
         "lastUpdatedTime" : "2021-12-16 10:16:25",
         "name" : "example_name1",
         "tags" : [
            "123",
            "456"
         ]
      },
      {
         "creationTime" : "2021-12-12 02:27:05",
         "description" : null,
         "lastUpdatedTime" : "2021-12-14 12:49:33",
         "name" : "register",
         "tags" : []
      }
   ],
   "status" : "OK",
   "success" : true
}
```

### Get a registered model
`GET /api/v1/registered-model/{name}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/registered-model/example_name
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "Get the registered model instance",
   "result" : {
      "creationTime" : "2021-12-16 10:14:06",
      "description" : "example_description",
      "lastUpdatedTime" : "2021-12-16 10:14:06",
      "name" : "example_name",
      "tags" : [
         "123",
         "456"
      ]
   },
   "status" : "OK",
   "success" : true
}
```

### Patch a registered model
`PATCH /api/v1/registered-model/{name}`

**Example Request:**
```sh
curl -X PATCH -H "Content-Type: application/json" -d '
{
    "name": "new_name",
    "description": "new_description"
}' http://127.0.0.1:32080/api/v1/registered-model/example_name
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "Update the registered model instance",
   "result" : null,
   "status" : "OK",
   "success" : true
}
```

## Delete a registered model
`DELETE /api/v1/registered-model/{name}`

**Example Request**
```sh
curl -X DELETE http://127.0.0.1:32080/api/v1/registered-model/example_name
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "Delete the registered model instance",
   "result" : null,
   "status" : "OK",
   "success" : true
}
```

## Create a registered model tag
`POST /api/v1/registered-model/tag?name={name}&tag={tag}`

**Example Request**
```sh
curl -X POST http://127.0.0.1:32080/api/v1/registered-model/tag?name=example_name&tag=789
```

**Example Response:**
```json
{
    "status":"OK",
    "code":200,
    "success":true,
    "message":"Create a registered model tag instance",
    "result":null,
    "attributes":{}
}
```

## Delete a registered model tag
`DELETE /api/v1/registered-model/tag?name={name}&tag={tag}`

**Example Request**
```sh
curl -X DELETE http://127.0.0.1:32080/api/v1/registered-model/tag?name=example_name&tag=789
```

**Example Response:**
```json
{
   "attributes" : {},
   "code" : 200,
   "message" : "Delete a registered model tag instance",
   "result" : null,
   "status" : "OK",
   "success" : true
}
```
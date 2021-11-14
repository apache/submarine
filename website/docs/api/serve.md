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

> Note: The Serv API is in the alpha stage which is subjected to incompatible changes in future releases.

## Create a TensorFlow model serve
`POST /api/v1/serve`

**Example Request**
> Make sure there is a model named `simple` with version `1` in the database.

```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "modelName": "simple", 
  "modelVersion":1, 
  "serveType":"tensorflow"
}
' http://127.0.0.1:32080/api/v1/serve
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":"Create a serve instance",
  "result":{"url":null},
  "attributes":{}
}
```

## Delete the TensorFlow model serve
`DELETE /api/v1/serve`

**Example Request**
```sh
curl -X DELETE -H "Content-Type: application/json" -d '
{
  "modelName": "simple", 
  "modelVersion":1, 
  "serveType":"tensorflow"
}
' http://127.0.0.1:32080/api/v1/serve
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":"Delete the model serve instance",
  "result":null,
  "attributes":{}
}
```

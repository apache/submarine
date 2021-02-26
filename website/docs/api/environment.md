---
title: Environment REST API
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

> Note: The Environment API is in the alpha stage which is subjected to incompatible changes in
> future releases.

## Create Environment
`POST /api/v1/environment`

**Example Request**
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "name": "my-submarine-env",
  "dockerImage" : "continuumio/anaconda3",
  "kernelSpec" : {
    "name" : "team_default_python_3.7",
    "channels" : ["defaults"],
    "dependencies" : 
      ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
      "alabaster=0.7.12=py37_0",
      "anaconda=2020.02=py37_0",
      "anaconda-client=1.7.2=py37_0",
      "anaconda-navigator=1.9.12=py37_0"]
  }
}
' http://127.0.0.1:8080/api/v1/environment
```

**Example Response:**
```json
{
  "status": "OK",
  "code": 200,
  "result": {
    "environmentId": "environment_1586156073228_0001",
    "environmentSpec": {
      "name": "my-submarine-env",
      "dockerImage" : "continuumio/anaconda3",
      "kernelSpec" : {
        "name" : "team_default_python_3.7",
        "channels" : ["defaults"],
        "dependencies" : 
          ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
          "alabaster=0.7.12=py37_0",
          "anaconda=2020.02=py37_0",
          "anaconda-client=1.7.2=py37_0",
          "anaconda-navigator=1.9.12=py37_0"]
      }
    }
  }
}
```

### List environment
`GET /api/v1/environment`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/environment
```

**Example Response:**
```json
{
  "status": "OK",
  "code": 200,
  "result": [
  {
    "environmentId": "environment_1586156073228_0001",
    "environmentSpec": {
      "name": "my-submarine-env",
      "dockerImage" : "continuumio/anaconda3",
      "kernelSpec" : {
        "name" : "team_default_python_3.7",
        "channels" : ["defaults"],
        "dependencies" : 
          ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
          "alabaster=0.7.12=py37_0",
          "anaconda=2020.02=py37_0",
          "anaconda-client=1.7.2=py37_0",
          "anaconda-navigator=1.9.12=py37_0"]
      }
    }
  },
  {
    "environmentId": "environment_1586156073228_0002",
    "environmentSpec": {
      "name": "my-submarine-env-2",
      "dockerImage" : "continuumio/miniconda",
      "kernelSpec" : {
        "name" : "team_miniconda_python_3.7",
        "channels" : ["defaults"],
        "dependencies" : 
          ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0"]
      }
    }
  }
  ]
}
```

### Get environment
`GET /api/v1/environment/{name}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/environment/my-submarine-env
```

**Example Response:**
```json
{
  "status": "OK",
  "code": 200,
  "result": {
    "environmentId": "environment_1586156073228_0001",
    "environmentSpec": {
      "name": "my-submarine-env",
      "dockerImage" : "continuumio/anaconda3",
      "kernelSpec" : {
        "name" : "team_default_python_3.7",
        "channels" : ["defaults"],
        "dependencies" : 
          ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
          "alabaster=0.7.12=py37_0",
          "anaconda=2020.02=py37_0",
          "anaconda-client=1.7.2=py37_0",
          "anaconda-navigator=1.9.12=py37_0"]
      }
    }
  }
}
```

### Patch environment
`PATCH /api/v1/environment/{name}`

**Example Request:**
```sh
curl -X PATCH -H "Content-Type: application/json" -d '
{
  "name": "my-submarine-env",
  "dockerImage" : "continuumio/anaconda3",
  "kernelSpec" : {
    "name" : "team_default_python_3.7_updated",
    "channels" : ["defaults"],
    "dependencies" : 
      ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
      "alabaster=0.7.12=py37_0"]
  }
}
' http://127.0.0.1:8080/api/v1/environment/my-submarine-env
```

**Example Response:**
```json
{
  "status": "OK",
  "code": 200,
  "success": true,
  "result": {
    "environmentId": "environment_1586156073228_0001",
    "environmentSpec": {
      "name": "my-submarine-env",
      "dockerImage" : "continuumio/anaconda3",
      "kernelSpec" : {
        "name" : "team_default_python_3.7_updated",
        "channels" : ["defaults"],
        "dependencies" : 
          ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
          "alabaster=0.7.12=py37_0"]
      }
    }
  }
}
```
> dockerImage, "name" (of kernelSpec), "channels", "dependencies" etc can be updated using this API.
"name" of EnvironmentSpec is not supported.

### Delete environment
`GET /api/v1/environment/{name}`

**Example Request:**
```sh
curl -X DELETE http://127.0.0.1:8080/api/v1/environment/my-submarine-env
```

**Example Response:**
```json
{
  "status": "OK",
  "code": 200,
  "result": {
    "environmentId": "environment_1586156073228_0001",
    "environmentSpec": {
      "name": "my-submarine-env",
      "dockerImage" : "continuumio/anaconda3",
      "kernelSpec" : {
        "name" : "team_default_python_3.7_updated",
        "channels" : ["defaults"],
        "dependencies" : 
          ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
          "alabaster=0.7.12=py37_0"]
      }
    }
  }
}
```
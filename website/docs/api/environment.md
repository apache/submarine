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

:::caution
The Environment API is in the alpha stage which is subjected to incompatible changes in future releases.
:::
## Create Environment
`POST /api/v1/environment`

### Parameters

Put EnvironmentSpec in request body.
#### **EnvironmentSpec**

| Field Name  | Type       | Description                 | Required |
| ----------- | ---------- | --------------------------- | :------: |
| name        | String     | Environment name.           |    o     |
| dockerImage | String     | Docker image name.          |    o     |
| kernelSpec  | KernelSpec | Environment spec.           |    o     |
| description | String     | Description of environment. |    x     |

#### **KernelSpec**

| Field Name        | Type          | Description                        | Required |
| ----------------- | ------------- | ---------------------------------- | :------: |
| name              | String        | Kernel name.                       |    o     |
| channels          | List<String\> | Names of the channels.             |    o     |
| condaDependencies | List<String\> | List of kernel conda dependencies. |    o     |
| pipDependencies   | List<String\> | List of kernel pip dependencies.   |    o     |

### Example
<details>
<summary>Example Request</summary>
<div>

```shell
curl -X POST -H "Content-Type: application/json" -d '
{
  "name": "my-submarine-env",
  "dockerImage" : "continuumio/anaconda3",
  "kernelSpec" : {
    "name" : "team_default_python_3.7",
    "channels" : ["defaults"],
    "condaDependencies" : 
      ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
      "alabaster=0.7.12=py37_0",
      "anaconda=2020.02=py37_0",
      "anaconda-client=1.7.2=py37_0",
      "anaconda-navigator=1.9.12=py37_0"],
    "pipDependencies" : 
      ["apache-submarine==0.7.0",
      "pyarrow==0.17.0"]
  }
}
' http://127.0.0.1:32080/api/v1/environment
```
</div>
</details>
<details>
<summary>Example Response</summary>
<div>

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
        "condaDependencies" : 
          ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
          "alabaster=0.7.12=py37_0",
          "anaconda=2020.02=py37_0",
          "anaconda-client=1.7.2=py37_0",
          "anaconda-navigator=1.9.12=py37_0"],
        "pipDependencies" : 
          ["apache-submarine==0.7.0",
          "pyarrow==0.17.0"]
      }
    }
  }
}
```
</div>
</details>


## List environment
`GET /api/v1/environment`

### Example
<details>
<summary>Example Request</summary>
<div>

```shell
curl -X GET http://127.0.0.1:32080/api/v1/environment
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
  "result":[
    {
      "environmentId":"environment_1600862964725_0002",
      "environmentSpec":{
        "name":"notebook-gpu-env",
        "dockerImage":"apache/submarine:jupyter-notebook-gpu-0.7.0-SNAPSHOT",
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
      }
    },
    {
      "environmentId":"environment_1647192232698_0003",
      "environmentSpec":{
        "name":"my-submarine-env",
        "dockerImage":"continuumio/anaconda3",
        "kernelSpec":{
          "name":"team_default_python_3.7",
          "channels":[
            "defaults"
          ],
          "condaDependencies":[
            "_ipyw_jlab_nb_ext_conf\u003d0.1.0\u003dpy37_0",
            "alabaster\u003d0.7.12\u003dpy37_0",
            "anaconda\u003d2020.02\u003dpy37_0",
            "anaconda-client\u003d1.7.2\u003dpy37_0",
            "anaconda-navigator\u003d1.9.12\u003dpy37_0"
          ],
          "pipDependencies":[
            "apache-submarine\u003d\u003d0.7.0",
            "pyarrow\u003d\u003d0.17.0"
          ]
        },
        "description":null,
        "image":null
      }
    },
    {
      "environmentId":"environment_1600862964725_0001",
      "environmentSpec":{
        "name":"notebook-env",
        "dockerImage":"apache/submarine:jupyter-notebook-0.7.0-SNAPSHOT",
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
      }
    }
  ],
  "attributes":{}
}
```
</div>
</details>


## Get environment
`GET /api/v1/environment/{name}`

### Parameters

| Field Name | Type   | In   | Description       | Required |
| ---------- | ------ | ---- | ----------------- | :------: |
| name       | String | path | Environment name. |    o     |

### Example
<details>
<summary>Example Request</summary>
<div>

```shell
curl -X GET http://127.0.0.1:32080/api/v1/environment/my-submarine-env
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
    "environmentId":"environment_1647192232698_0003",
    "environmentSpec":{
      "name":"my-submarine-env",
      "dockerImage":"continuumio/anaconda3",
      "kernelSpec":{
        "name":"team_default_python_3.7",
        "channels":[
          "defaults"
        ],
        "condaDependencies":[
          "_ipyw_jlab_nb_ext_conf\u003d0.1.0\u003dpy37_0",
          "alabaster\u003d0.7.12\u003dpy37_0",
          "anaconda\u003d2020.02\u003dpy37_0",
          "anaconda-client\u003d1.7.2\u003dpy37_0",
          "anaconda-navigator\u003d1.9.12\u003dpy37_0"
        ],
        "pipDependencies":[
          "apache-submarine\u003d\u003d0.7.0",
          "pyarrow\u003d\u003d0.17.0"
        ]
      },
      "description":null,
      "image":null
    }
  },
  "attributes":{}
}
```
</div>
</details>


## Patch environment
`PATCH /api/v1/environment/{name}`

### Parameters

| Field Name  | Type       | In            | Description                                         | Required |
| ----------- | ---------- | ------------- | --------------------------------------------------- | :------: |
| name        | String     | path and body | Environment name.                                   |    o     |
| dockerImage | String     | body          | Docker image name.                                  |    o     |
| kernelSpec  | KernelSpec | body          | Environment spec.                                   |    o     |
| description | String     | body          | Description of environment. This field is optional. |    x     |

### Example

<details>
<summary>Example Request</summary>
<div>

```shell
curl -X PATCH -H "Content-Type: application/json" -d '
{
  "name": "my-submarine-env",
  "dockerImage" : "continuumio/anaconda3",
  "kernelSpec" : {
    "name" : "team_default_python_3.7_updated",
    "channels" : ["defaults"],
    "condaDependencies" : 
      ["_ipyw_jlab_nb_ext_conf=0.1.0=py37_0",
      "alabaster=0.7.12=py37_0"],
    "pipDependencies" : 
      []
  }
}
' http://127.0.0.1:32080/api/v1/environment/my-submarine-env
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
    "environmentId":"environment_1647192232698_0004",
    "environmentSpec":{
      "name":"my-submarine-env",
      "dockerImage":"continuumio/anaconda3",
      "kernelSpec":{
        "name":"team_default_python_3.7_updated",
        "channels":[
          "defaults"
        ],
        "condaDependencies":[
          "_ipyw_jlab_nb_ext_conf\u003d0.1.0\u003dpy37_0",
          "alabaster\u003d0.7.12\u003dpy37_0"
        ],
        "pipDependencies":[]
      },
      "description":null,
      "image":null
    }
  },
  "attributes":{}
}
```
</div>
</details>

:::note
dockerImage, "name" (of kernelSpec), "channels", "condaDependencies", "pipDependencies" etc can be updated using this API.
"name" of environmentSpec is not supported.
:::

## Delete environment
`GET /api/v1/environment/{name}`

### Parameters

| Field Name | Type   | In   | Description       | Required |
| ---------- | ------ | ---- | ----------------- | :------: |
| name       | String | path | Environment name. |    o     |

### Example
<details>
<summary>Example Request</summary>
<div>

```shell
curl -X DELETE http://127.0.0.1:32080/api/v1/environment/my-submarine-env
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
    "environmentId":"environment_1647192232698_0003",
    "environmentSpec":{
      "name":"my-submarine-env",
      "dockerImage":"continuumio/anaconda3",
      "kernelSpec":{
        "name":"team_default_python_3.7",
        "channels":[
          "defaults"
        ],
        "condaDependencies":[
          "_ipyw_jlab_nb_ext_conf\u003d0.1.0\u003dpy37_0",
          "alabaster\u003d0.7.12\u003dpy37_0",
          "anaconda\u003d2020.02\u003dpy37_0",
          "anaconda-client\u003d1.7.2\u003dpy37_0",
          "anaconda-navigator\u003d1.9.12\u003dpy37_0"
        ],
        "pipDependencies":[
          "apache-submarine\u003d\u003d0.7.0",
          "pyarrow\u003d\u003d0.17.0"
        ]
      },
      "description":null,
      "image":null
    }
  },
  "attributes":{}
}
```

</div>
</details>

---
title: Submarine Client
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

## class SubmarineClient()

Client of submarine to log metric/param, save model and create/delete serve.


#### `log_metric(job_id, key, value, worker_index, timestamp, step) -> None`

Log a single key-value metric with job id and worker index. The value must always be a number.

|    Param     |   Type   | Description                                                                     | Default Value |
| :----------: | :------: | ------------------------------------------------------------------------------- | :-----------: |
|    job_id    |  String  | The job name to which the metric should be logged.                              |       x       |
|     key      |  String  | Metric name.                                                                    |       x       |
|    value     |  Float   | Metric worker_index.                                                            |       x       |
| worker_index |  String  | Parameter worker_index.                                                         |       x       |
|  timestamp   | Datetime | Time when this metric was calculated. Defaults to the current system time.      |       datetime.now()       |
|     step     | Integer  | A single integer step at which to log the specified Metrics, by default it's 0. |       0       |

<br />

#### `log_param(job_id, key, value, worker_index) -> None`

Log a single key-value parameter with job id and worker index. The key and value are both strings.

|    Param     |  Type  | Description                                           | Default Value |
| :----------: | :----: | ----------------------------------------------------- | :-----------: |
|    job_id    | String | The job name to which the parameter should be logged. |       x       |
|     key      | String | Parameter name.                                       |       x       |
|    value     | String | Parameter value.                                      |       x       |
| worker_index | String | Parameter worker_index.                               |       x       |

<br />

#### `save_model(model, model_type, registered_model_name, input_dim, output_dim) -> None`

Save a model into the minio pod.

|         Param         |     Type      | Description                                                                               | Default Value |
| :-------------------: | :-----------: | ----------------------------------------------------------------------------------------- | :-----------: |
|         model         |    Object     | Model artifact.                                                                           |       x       |
|      model_type       |    String     | Version of a registered model.                                                            |       x       |
| registered_model_name |    String     | If it is not `None`, the model will be registered into the model registry with this name. |     None      |
|       input_dim       | List<String\> | The input dimension of the model.                                                         |     None      |
|      output_dim       | List<String\> | The output dimension of the model.                                                        |     None      |

<br />

#### `create_serve(self, model_name, model_version, async_req = True) -> dict`

Create serve of a model through Seldon Core.

|     Param     |  Type   | Description                     | Default Value |
| :-----------: | :-----: | ------------------------------- | :-----------: |
|  model_name   | String  | Name of a registered model.     |       x       |
| model_version | Integer | Version of a registered model.  |       x       |
|   async_req   | Boolean | Execute request asynchronously. |     True      |

<br />

**Returns**
Return a dictionary with inference url.
#### `delete_serve(self, model_name, model_version, async_req) -> None`

Delete a serving model.

|     Param     |  Type   | Description                     | Default Value |
| :-----------: | :-----: | ------------------------------- | :-----------: |
|  model_name   | String  | Name of a registered model.     |       x       |
| model_version | Integer | Version of a registered model.  |       x       |
|   async_req   | Boolean | Execute request asynchronously. |     True      |

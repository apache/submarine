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

### `log_metric(job_id: str, key: str, value: float, worker_index: str, timestamp: datetime = None, step: int = None) -> None`

Log a single key-value metric with job id and worker index. The value must always be a number.

> **Parameters**
  - **job_id**: The job name to which the metric should be logged.
  - **key** - Metric name.
  - **value** - Metric value.
  - **worker_index** - Metric worker_index.
  - **timestamp** - Time when this metric was calculated. Defaults to the current system time.
  - **step** - A single integer step at which to log the specified Metrics, by default it's 0.

### `log_param(job_id: str, key: str, value: str, worker_index: str) -> None`

Log a single key-value parameter with job id and worker index. The key and value are both strings.

> **Parameters**
  - **job_id** - The job name to which the parameter should be logged.
  - **key** - Parameter name.
  - **value** - Parameter value.
  - **worker_index** - Parameter worker_index.


### `save_model(model, model_type: str, registered_model_name: str = None, input_dim: list = None, output_dim: list = None) -> None`

Save a model into the minio pod.

> **Parameters**
  - **model** - Model artifact.
  - **model_type** - The type of model. Only support `pytorch` and `tensorflow`.
  - **registered_model_name** - If it is not `None`, the model will be registered into the model registry with this name.
  - **input_dim** - The input dimension of the model.
  - **output_dim** - The output dimension of the model.

### `create_serve(self, model_name: str, model_version: int, async_req: bool = True)`

Create serve of a model through Seldon Core.

> **Parameters**
  - **model_name** - Name of a registered model
  - **model_version**: Version of a registered model
  - **async_req** - Execute request asynchronously

### `delete_serve(self, model_name: str, model_version: int, async_req: bool = True)`

Delete a serving model.

> **Parameters**
  - **model_name** - Name of a registered model
  - **model_version**: Version of a registered model
  - **async_req** - Execute request asynchronously

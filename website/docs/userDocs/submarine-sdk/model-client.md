---
title: Model Client
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

## class ModelClient()

The submarine ModelsClient provides a high-level API for logging metrics / parameters and managing models.

### `ModelsClient(tracking_uri=None, registry_uri=None)->ModelsClient`

Initialize a `ModelsClient` instance.

> **Parameters**
  - **tracking_uri**: If run in Submarine, you do not need to specify it. Otherwise, specify the external tracking_uri.
  - **registry_uri**:  If run in Submarine, you do not need to specify it. Otherwise, specify the external registry_uri.

> **Returns**
  - ModelsClient instance

Example

```python
from submarine import ModelsClient

modelClient = ModelsClient(tracking_uri="0.0.0.0:4000", registry_uri="0.0.0.0:5000")
```
### `ModelsClient.start()->[Active Run]`

For details of [Active Run](https://mlflow.org/docs/latest/_modules/mlflow/tracking/fluent.html#ActiveRun)

Start a new Mlflow run, and direct the logging of the artifacts and metadata to the Run named "worker_i" under Experiment "job_id". If in distributed training, worker and job id would be parsed from environment variable. If in local traning, worker and job id will be generated.

> **Returns**
  - Active Run

### `ModelsClient.log_param(key, value)->None`

Log parameter under the current run.

> **Parameters**
  - **key** – Parameter name
  - **value** – Parameter value

Example

```python
from submarine import ModelsClient

modelClient = ModelsClient()
with modelClient.start() as run:
  modelClient.log_param("learning_rate", 0.01)
```

### `ModelsClient.log_params(params)->None`

Log a batch of params for the current run.

> **Parameters**
  - **params** – Dictionary of param_name: String -> value

Example

```python
from submarine import ModelsClient

params = {"learning_rate": 0.01, "n_estimators": 10}

modelClient = ModelsClient()
with modelClient.start() as run:
  modelClient.log_params(params)
```

### `ModelsClient.log_metric(self, key, value, step=None)->None`

Log a metric under the current run.

> **Parameters**
  - **key** – Metric name (string).
  - **value** – Metric value (float).
  - **step** – Metric step (int). Defaults to zero if unspecified.

Example

```python
from submarine import ModelsClient

modelClient = ModelsClient()
with modelClient.start() as run:
  modelClient.log_metric("mse", 2500.00)
```

### `ModelsClient.log_metrics(self, metrics, step=None)->None`

Log multiple metrics for the current run.

> **Parameters**
  - **metrics** – Dictionary of metric_name: String -> value: Float.
  - **step** – A single integer step at which to log the specified Metrics. If unspecified, each metric is logged at step zero.

Example

```python
from submarine import ModelsClient

metrics = {"mse": 2500.00, "rmse": 50.00}

modelClient = ModelsClient()
with modelClient.start() as run:
  modelClient.log_metrics(metrics)
```

### `(Beta) ModelsClient.save_model(self, model_type, model, artifact_path, registered_model_name=None)`

Save model to model registry.
### `(Beta) ModelsClient.load_model(self, name, version)->mlflow.pyfunc.PyFuncModel`

Load a model from model registry.
### `(Beta) ModelsClient.update_model(self, name, new_name)->None`

Update a model by new name.

### `(Beta) ModelsClient.delete_model(self, name, version)->None`

Delete a model in model registry.

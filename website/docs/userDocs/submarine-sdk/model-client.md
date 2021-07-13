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

The submarine ModelsClient provides a high-level API for managing and serving your models.

## ModelsClient.start()->[Active Run](https://mlflow.org/docs/latest/_modules/mlflow/tracking/fluent.html#ActiveRun)

1. Start a new Mlflow run
2. Direct the logging of the artifacts and metadata to the Run named "worker_i" under Experiment "job_id"
3. If in distributed training, worker and job id would be parsed from environment variable
4. If in local traning, worker and job id will be generated.
   :return: Active Run

## ModelsClient.log_param(key, value)->None

Log parameter under the current run.

- ### Parameters
  - **key** – Parameter name (string)
  - **value** – Parameter value (string, but will be string-ified if not)
- ### example

```
from submarine import ModelsClient

periscope = ModelsClient()
with periscope.start() as run:
  periscope.log_param("learning_rate", 0.01)
```

## ModelsClient.log_params(params)->None

Log a batch of params for the current run.

- ### Parameters

  - **params** – Dictionary of param_name: String -> value: (String, but will be string-ified if not)

- ### example

```
from submarine import ModelsClient

params = {"learning_rate": 0.01, "n_estimators": 10}

periscope = ModelsClient()
with periscope.start() as run:
  periscope.log_params(params)
```

## ModelsClient.log_metric(self, key, value, step=None)->None

Log a metric under the current run.

- ### Parameters

  - **key** – Metric name (string).
  - **value** – Metric value (float). Note that some special values such as +/- Infinity may be replaced by other values depending on the store. For example, the SQLAlchemy store replaces +/- Infinity with max / min float values.
  - **step** – Metric step (int). Defaults to zero if unspecified.

- ### example

```
from submarine import ModelsClient

periscope = ModelsClient()
with periscope.start() as run:
  periscope.log_metric("mse", 2500.00)
```

## ModelsClient.log_metrics(self, metrics, step=None)->None

Log multiple metrics for the current run.

- ### Parameters

  - **metrics** – Dictionary of metric_name: String -> value: Float. Note that some special values such as +/- Infinity may be replaced by other values depending on the store. For example, sql based store may replace +/- Infinity with max / min float values.
  - **step** – A single integer step at which to log the specified Metrics. If unspecified, each metric is logged at step zero.

- ### example

```
from submarine import ModelsClient

metrics = {"mse": 2500.00, "rmse": 50.00}

periscope = ModelsClient()
with periscope.start() as run:
  periscope.log_metrics(metrics)
```

## ModelsClient.load_model(self, name, version)->[ mlflow.pyfunc.PyFuncModel](https://mlflow.org/docs/latest/_modules/mlflow/pyfunc.html#PyFuncModel)

Load a model stored in models Python function format with specific name and version.

- ### Parameters
  - **name** – Name of the containing registered model.(string).
  - **version** – Version number of the model version.(string).

## ModelsClient.update_model(self, name, new_name)->None

Update registered model name.

- ### Parameters
  - **name** – Name of the registered model to update(string).
  - **new name** – New proposed name for the registered model(string).

## ModelsClient.delete_model(self, name, version)->None

Delete model version in backend.

- ### Parameters
  - **name** – Name of the containing registered model.(string).
  - **version** – Version number of the model version.(string).

## ModelsClient.save_model(self, model_type, model, artifact_path, registered_model_name=None)

Beta: Save model to server for managment and servering.

- ### Parameters
  - **name** – Name of the containing registered model.(string).
  - **version** – Version number of the model version.(string).

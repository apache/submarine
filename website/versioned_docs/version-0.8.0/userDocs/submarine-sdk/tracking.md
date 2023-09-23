---
title: Tracking
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

It helps developers use submarine's internal data caching,
data exchange, and task tracking capabilities to more efficiently improve the
development and execution of machine learning productivity

- Allow data scientist to track distributed ML experiment
- Support store ML parameters and metrics in Submarine-server
- Support hdfs, S3 and mysql (Currently we only support mysql)

### Functional api

#### `submarine.get_tracking_uri() -> str`

Get the tracking URI. If none has been specified, check the environmental variables. If uri is still none, return the default submarine jdbc url.

**Returns**

The tracking URI.
<br />

#### `submarine.set_tracking_uri(uri) -> None`

set the tracking URI. You can also set the SUBMARINE_TRACKING_URI environment variable to have Submarine find a URI from there. The URI should be database connection string.

| Param |  Type  | Description                                                                                                                                                                                                                                                                                                                                              | Default Value |
| :---: | :----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------: |
|  uri  | String | Submarine record data to Mysql server. The database URL is expected in the format ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>``.By default it's `mysql+pymysql://submarine:password@submarine-database:3306/submarine`. More detail : [SQLAlchemy docs](https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls) |       x       |

<br />

#### `submarine.log_param(key: str, value: str) -> None`

log a single key-value parameter. The key and value are both strings.

| Param |  Type  | Description      | Default Value |
| :---: | :----: | ---------------- | :-----------: |
|  key  | String | Parameter name.  |       x       |
| value | String | Parameter value. |       x       |

<br />

#### `submarine.log_metric(key, value, step=0) -> None`
log a single key-value metric. The value must always be a number.

| Param |  Type   | Description                                                  | Default Value |
| :---: | :-----: | ------------------------------------------------------------ | :-----------: |
|  key  | String  | Metric name.                                                 |       x       |
| value |  Float  | Metric value.                                                |       x       |
| step  | Integer | A single integer step at which to log the specified Metrics. |       0       |

<br />

#### `submarine.save_model(model_type, model, registered_model_name, input_dim, output_dim) -> None`

Save a model into the minio pod.

|         Param         |      Type      | Description                                                                               | Default Value |
| :-------------------: | :------------: | ----------------------------------------------------------------------------------------- | :-----------: |
|      model_type       |     String     | The type of model. Only support `pytorch` and `tensorflow`.                               |       x       |
|         model         |     Object     | Model artifact.                                                                           |       x       |
| registered_model_name |     String     | If it is not `None`, the model will be registered into the model registry with this name. |     None      |
|       input_dim       | List<Integer\> | The input dimension of the model.                                                         |     None      |
|      output_dim       | List<Integer\> | The output dimension of the model.                                                        |     None      |

<br />


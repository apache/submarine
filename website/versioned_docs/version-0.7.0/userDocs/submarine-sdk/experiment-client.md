---
title: Experiment Client
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

## class ExperimentClient()

Client of a submarine server that creates and manages experients and logs.

#### `create_experiment(experiment_spec) -> dict`

Create an experiment.

|      Param      | Type  | Description                                                                                                    | Default Value |
| :-------------: | :---: | -------------------------------------------------------------------------------------------------------------- | :-----------: |
| experiment_spec | Dict  | Submarine experiment spec. More detailed information can be found at [Experiment API](../../api/experiment.md) |       x       |

**Returns**

The detailed info about the submarine experiment.


**Example**

```python
from submarine import *
client = ExperimentClient()
client.create_experiment({
  "meta": {
    "name": "tf-mnist-json",
    "namespace": "default",
    "framework": "TensorFlow",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "environment": {
    "image": "apache/submarine:tf-mnist-with-summaries-1.0"
  },
  "spec": {
    "Ps": {
      "replicas": 1,
      "resources": "cpu=1,memory=1024M"
    },
    "Worker": {
      "replicas": 1,
      "resources": "cpu=1,memory=1024M"
    }
  }
})
```
<br />

#### `patch_experiment(id, experiment_spec) -> dict`

Patch an experiment.

|      Param      |  Type  | Description                                                                                                                                  | Default Value |
| :-------------: | :----: | -------------------------------------------------------------------------------------------------------------------------------------------- | :-----------: |
|       id        | String | Submarine experiment id.                                                                                                                     |       x       |
| experiment_spec |  Dict  | Submarine experiment spec. More detailed information of Submarine experiment spec can be found at [Experiment API](../../api/experiment.md). |       x       |

**Returns**

The detailed info about the submarine experiment.

**Example**

```python
client.patch_experiment("experiment_1626160071451_0008", {
  "meta": {
    "name": "tf-mnist-json",
    "namespace": "default",
    "framework": "TensorFlow",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "environment": {
    "image": "apache/submarine:tf-mnist-with-summaries-1.0"
  },
  "spec": {
    "Worker": {
      "replicas": 2,
      "resources": "cpu=1,memory=1024M"
    }
  }
})
```
<br />

#### `get_experiment(id) -> dict`

Get the experiment's detailed info by id.

| Param |  Type  | Description              | Default Value |
| :---: | :----: | ------------------------ | :-----------: |
|  id   | String | Submarine experiment id. |       x       |

**Returns**

The detailed info about the submarine experiment.

**Example**

```python
experiment = client.get_experiment("experiment_1626160071451_0008")
```
<br />

#### `list_experiments(status) -> list[dict]`

List all experiment for the user.

| Param  |     Type      | Description                                     | Default Value |
| :----: | :-----------: | ----------------------------------------------- | :-----------: |
| status | Optional[str] | Accepted, Created, Running, Succeeded, Deleted. |     None      |

**Returns**

List of submarine experiments.

**Example**

```python
experiments = client.list_experiments()
```
<br />

#### `delete_experiment(id) -> dict`

Delete the submarine experiment.

| Param |  Type  | Description              | Default Value |
| :---: | :----: | ------------------------ | :-----------: |
|  id   | String | Submarine experiment id. |       x       |

**Returns**

The detailed info about the deleted submarine experiment.

**Example**

```python
client.delete_experiment("experiment_1626160071451_0008")
```
<br />

#### `get_log(id, onlyMaster)`

Print training logs of all pod of the experiment.
By default print all the logs of Pod.

|   Param    |      Type      | Description                                                                                  | Default Value |
| :--------: | :------------: | -------------------------------------------------------------------------------------------- | :-----------: |
|     id     |     String     | Submarine experiment id.                                                                     |       x       |
| onlyMaster | Optional[bool] | By default include pod log of "master" which might be Tensorflow PS/Chief or PyTorch master. |       x       |

**Return**
  - The info of pod logs

**Example**

```python
client.get_log("experiment_1626160071451_0009")
```
<br />

#### `list_log(status)`

List experiment log.

| Param  |  Type  | Description                                     | Default Value |
| :----: | :----: | ----------------------------------------------- | :-----------: |
| status | String | Accepted, Created, Running, Succeeded, Deleted. |       x       |

**Returns**

List of submarine experiment logs.

Example

```python
logs = client.list_log("Succeeded")
```
<br />

#### `wait_for_finish(id, polling_interval)`

Waits until the experiment is finished or failed.

|      Param       |     Type      | Description                                                          | Default Value |
| :--------------: | :-----------: | -------------------------------------------------------------------- | :-----------: |
|        id        |    String     | Submarine experiment id.                                             |       x       |
| polling_interval | Optional[int] | How many seconds between two polls for the status of the experiment. |      10       |


**Returns**

Submarine experiment logs.

**Example**

```python
logs = client.wait_for_finish("experiment_1626160071451_0009", 5)
```
<br />
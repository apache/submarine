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

### `create_experiment(experiment_spec: json) -> dict`

Create an experiment.
> **Parameters**
  - **experiment_spec**: Submarine experiment spec. More detailed information can be found at [Experiment API](https://submarine.apache.org/docs/userDocs/api/experiment).
> Returns
  - The detailed info about the submarine experiment.

### `patch_experiment(id: str, experiment_spec: json) -> dict`

Patch an experiment.
> **Parameters**
  - **id**: Submarine experiment id. 
  - **experiment_spec**: Submarine experiment spec. More detailed information can be found at [Experiment API](https://submarine.apache.org/docs/userDocs/api/experiment).
> **Returns**
  - The detailed info about the submarine experiment.

### `get_experiment(id: str) -> dict`

Get the experiment's detailed info by id.
> **Parameters**
  - **id**: Submarine experiment id.

> **Returns**
  - The detailed info about the submarine experiment.

### `list_experiments(status: Optional[str]=None) -> list[dict]`

List all experiment for the user.
> **Parameters**
  - **status**: Accepted, Created, Running, Succeeded, Deleted.

> **Returns**
  - List of submarine experiments.

### `delete_experiment(id: str) -> dict`

Delete the submarine experiment.
> **Parameters**
  - **id**: Submarine experiment id.

> **Returns**
  - The detailed info about the deleted submarine experiment.

### `get_log(id: str, onlyMaster: Optional[bool]=False) -> dict`

Get training logs of all pod of the experiment.
By default get all the logs of Pod.

> **Parameters**
  - **id**: Submarine experiment id.
  - **onlyMaster**: By default include pod log of "master" which might be Tensorflow PS/Chief or PyTorch master.
> **Returns**
  - Submarine experiment pods' logs.

### `list_log(status: str) -> list[dict]`

List experiment log.
> **Parameters**
  - **status**: Accepted, Created, Running, Succeeded, Deleted.
> Returns
  - List of submarine experiment logs.

### `wait_for_finish(id: str, polling_interval: Optional[int]=10) -> dict`

Waits until the experiment is finished or failed.
> **Parameters**
  - **id**: Submarine experiment id.
  - **polling_interval**: How many seconds between two polls for the status of the experiment.
> **Returns**
  - Submarine experiment logs.

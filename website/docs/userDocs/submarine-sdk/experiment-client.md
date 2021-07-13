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

Example

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

Output

```python
{
  'experimentId': 'experiment_1626160071451_0008', 
  'name': 'tf-mnist-json', 'uid': '3513233e-33f2-4399-8fba-2a44ca2af730', 
  'status': 'Accepted', 
  'acceptedTime': '2021-07-13T21:29:33.000+08:00', 
  'createdTime': None, 
  'runningTime': None, 
  'finishedTime': None, 
  'spec': {
    'meta': {
      'name': 'tf-mnist-json', 
      'namespace': 'default', 
      'framework': 'TensorFlow', 
      'cmd': 'python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150', 
      'envVars': {'ENV_1': 'ENV1'}
    }, 
    'environment': {
      'name': None, 
      'dockerImage': None, 
      'kernelSpec': None, 
      'description': None, 
      'image': 'apache/submarine:tf-mnist-with-summaries-1.0'
    }, 
    'spec': {
      'Ps': {
        'replicas': 1, 
        'resources': 'cpu=1,memory=1024M', 
        'name': None, 
        'image': None, 
        'cmd': None, 
        'envVars': None, 
        'resourceMap': {'memory': '1024M', 'cpu': '1'}
      }, 
      'Worker': {
        'replicas': 1, 
        'resources': 'cpu=1,memory=1024M', 
        'name': None, 
        'image': None, 
        'cmd': None, 
        'envVars': None, 
        'resourceMap': {'memory': '1024M', 'cpu': '1'}
      }
    }, 
    'code': None
  }
}
```

### `patch_experiment(id: str, experiment_spec: json) -> dict`

Patch an experiment.
> **Parameters**
  - **id**: Submarine experiment id. 
  - **experiment_spec**: Submarine experiment spec. More detailed information can be found at [Experiment API](https://submarine.apache.org/docs/userDocs/api/experiment).
> **Returns**
  - The detailed info about the submarine experiment.

Example

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

### `get_experiment(id: str) -> dict`

Get the experiment's detailed info by id.
> **Parameters**
  - **id**: Submarine experiment id.

> **Returns**
  - The detailed info about the submarine experiment.

Example

```python
experiment = client.get_experiment("experiment_1626160071451_0008")
```

### `list_experiments(status: Optional[str]=None) -> list[dict]`

List all experiment for the user.
> **Parameters**
  - **status**: Accepted, Created, Running, Succeeded, Deleted.

> **Returns**
  - List of submarine experiments.

Example

```python
experiments = client.list_experiments()
```

### `delete_experiment(id: str) -> dict`

Delete the submarine experiment.
> **Parameters**
  - **id**: Submarine experiment id.

> **Returns**
  - The detailed info about the deleted submarine experiment.

Example

```python
client.delete_experiment("experiment_1626160071451_0008")
```

### `get_log(id: str, onlyMaster: Optional[bool]=False) -> None`

Print training logs of all pod of the experiment.
By default print all the logs of Pod.

> **Parameters**
  - **id**: Submarine experiment id.
  - **onlyMaster**: By default include pod log of "master" which might be Tensorflow PS/Chief or PyTorch master.

Example

```python
client.get_log("experiment_1626160071451_0009")
```

Output

```
The logs of Pod tf-mnist-json-2-ps-0:

The logs of Pod tf-mnist-json-2-worker-0:

```

### `list_log(status: str) -> list[dict]`

List experiment log.
> **Parameters**
  - **status**: Accepted, Created, Running, Succeeded, Deleted.
> **Returns**
  - List of submarine experiment logs.

Example

```python
logs = client.list_log("Succeeded")
```

Output

```python
[{'experimentId': 'experiment_1626160071451_0009',
  'logContent': 
  [{'podName': 'tf-mnist-json-2-ps-0', 'podLog': []},
   {'podName': 'tf-mnist-json-2-worker-0', 'podLog': []}]
}]
```

### `wait_for_finish(id: str, polling_interval: Optional[int]=10) -> dict`

Waits until the experiment is finished or failed.
> **Parameters**
  - **id**: Submarine experiment id.
  - **polling_interval**: How many seconds between two polls for the status of the experiment.
> **Returns**
  - Submarine experiment logs.

Example

```python
logs = client.wait_for_finish("experiment_1626160071451_0009", 5)
```

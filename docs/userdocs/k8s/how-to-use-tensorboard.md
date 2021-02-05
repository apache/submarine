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

# How to use tensorboard in submarine

## Write to LogDirs by the environment variable

### Usage

- `SUBMARINE_LOG_DIR`: This environment variable already exists in every experiment container. You just need to direct your logs to `$(SUBMARINE_LOG_DIR)` (**NOTICE: it is `()` not `{}`**), and you can inspect the process on the tensorboard webpage.

### Example

```
{
  "meta": {
    "name": "tensorflow-tensorboard-dist-mnist",
    "namespace": "default",
    "framework": "TensorFlow",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=$(SUBMARINE_LOG_DIR) --learning_rate=0.01 --batch_size=20",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "environment": {
    "image": "apache/submarine:tf-mnist-with-summaries-1.0"
  },
  "spec": {
    "Worker": {
      "replicas": 1,
      "resources": "cpu=1,memory=512M"
    }
  }
}

```

## Connect to the tensorboard webpaage

1. Open the experiment page in the workbench, and Click the `TensorBoard` button.

![](../../assets/tensorboard-experiment-page.png)

2. Inspect the process on tensorboard page.

![](../../assets/tensorboard-webpage.png)

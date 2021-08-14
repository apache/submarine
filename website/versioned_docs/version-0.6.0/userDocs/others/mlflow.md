---
title: MLflow UI
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

### Usage
MLflow UI shows the tracking result of the experiments. When we
use the log_param or log_metric in ModelClient API, we could view
the result in MLflow UI. Below is the example of the usage of MLflow
UI.

### Example
1. Run the following code in the cluster

```python
from submarine import ModelsClient
import random
import time

if __name__ == "__main__":
  modelClient = ModelsClient()
  with modelClient.start() as run:
      modelClient.log_param("learning_rate", random.random())
      for i in range(100):
        time.sleep(1)
        modelClient.log_metric("mse", random.random() * 100, i)
        modelClient.log_metric("acc", random.random(), i)
```

2. In the MLflow UI page, you can see the log_param and the log_metric
    result. You can also compare the training between different workers.

![](/img/mlflow-ui.png)


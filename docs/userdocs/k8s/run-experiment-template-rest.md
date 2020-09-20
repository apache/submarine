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

# Experiment Template REST API Reference

## Experiment Template Spec
The experiment is represented in [JSON](https://www.json.org) or [YAML](https://yaml.org) format.


### Use existing experiment template to create a experiment
`POST /api/v1/environment/{name}`

**Example Request:**
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
    "name": "tf-mnist",
    "params": {
        "learning_rate":"0.01", 
        "batch_size":"150", 
        "experiment_name":"newexperiment1"
    }
}
' http://127.0.0.1:8080/api/v1/experiment/my-tf-mnist-template
```

Register experiment template and more info see [Experiment Template API Reference](api/experiment-template.md).

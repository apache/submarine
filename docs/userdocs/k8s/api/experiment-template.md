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

# Experiment Template API Reference

> Note: The Experiment API is in the alpha stage which is subjected to incompatible changes in
> future releases.


Developers can register a parameterized experiment as an experiment template,
For example, if the developer wants to change the following "--learning_rate=0.1" to parameters.
```json
"experimentSpec": {
  "meta": {
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.1 --batch_size=150"
  }, 
  "...": "..."
}
```

They can use two curly braces as placeholders, the template format will be as
```json
"experimentSpec": {
  "meta": {
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{learning_rate}} --batch_size=150"
  }, 
  "...": "..."
}
```

The template parameters format will be as
```json
{
  "name": "learning_rate",
  "value": 0.1,
  "required": true,
  "description": "This is learning_rate of training."
}
```
name: placeholder name
value; default value
required: Indicates whether the user must enter parameters, when required is true, value can be null
description: Introduction of this parameter

Users can use existing experiment templates and adjust the default value to create experiments.
After the user submits the experiment template, the submarine server finds the corresponding template based on the name. And the template handler converts input parameters to an actual experiment, such as a distributed TF experiment.

The "replicas", "cpu", "memory" of resources will be automatically parameterized, so developers do not need to add them.
For example, if there are "Ps" and "Worker" under spec, the following parameters will be automatically appended.
```
spec.Ps.replicas
spec.Ps.resourceMap.cpu
spec.Ps.resourceMap.memory
spec.Worker.replicas
spec.Worker.resourceMap.cpu
spec.Worker.resourceMap.memory
```


## Create experiment template
`POST /api/v1/template`

**Example Request**
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "name": "my-tf-mnist-template",
  "author": "author",
  "description": "This is a template to run tf-mnist",
  "parameters": [{
      "name": "learning_rate",
      "value": 0.1,
      "required": true,
      "description": "This is learning_rate of training."
    },
    {
      "name": "batch_size",
      "value": 150,
      "required": true,
      "description": "This is batch_size of training."
    },
    {
      "name": "experiment_name",
      "value": "tf-mnist1",
      "required": true,
      "description": "the name of experiment."
    }
  ],
  "experimentSpec": {
    "meta": {
      "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{learning_rate}} --batch_size={{batch_size}}",
      "name": "{{experiment_name}}",
      "envVars": {
        "ENV1": "ENV1"
      },
      "framework": "TensorFlow",
      "namespace": "default"
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
    },
    "environment": {
      "image": "apache/submarine:tf-mnist-with-summaries-1.0"
    }
  }
}
' http://127.0.0.1:8080/api/v1/template
```


### List experiment template
`GET /api/v1/template`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/template
```

### Get experiment template
`GET /api/v1/template/{name}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/template/my-tf-mnist-template
```


### Patch template
`PATCH /api/v1/template/{name}`
```sh
curl -X PATCH -H "Content-Type: application/json" -d '
{
  "name": "my-tf-mnist-template",
  "author": "author-new",
  "description": "This is a template to run tf-mnist",
  "parameters": [{
      "name": "learning_rate",
      "value": 0.1,
      "required": true,
      "description": "This is learning_rate of training."
    },
    {
      "name": "batch_size",
      "value": 150,
      "required": true,
      "description": "This is batch_size of training."
    },
    {
      "name": "experiment_name",
      "value": "tf-mnist1",
      "required": true,
      "description": "the name of experiment."
    }
  ],
  "experimentSpec": {
    "meta": {
      "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{learning_rate}} --batch_size={{batch_size}}",
      "name": "{{experiment_name}}",
      "envVars": {
        "ENV1": "ENV1"
      },
      "framework": "TensorFlow",
      "namespace": "default"
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
    },
    "environment": {
      "image": "apache/submarine:tf-mnist-with-summaries-1.0"
    }
  }
}
' http://127.0.0.1:8080/api/v1/template/my-tf-mnist-template
```

> "description", "parameters", "experimentSpec", "author" etc can be updated using this API.
"name" of experiment template is not supported.



### Delete template
`GET /api/v1/template/{name}`

**Example Request:**
```sh
curl -X DELETE http://127.0.0.1:8080/api/v1/template/my-tf-mnist-template
```


### Use template to create a experiment
`POST /api/v1/experiment/{template_name}`

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

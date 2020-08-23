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

# Use Experiment Template Guide

The {{name}} variable in "experimentSpec" will be replace by the parameters value.

JSON Format example:
```json
{
  "name": "tf-mnist-test",
  "author": "author",
  "description": "This is a template to run tf-mnist\n",
  "parameters": [
    {
      "name": "training.learning_rate",
      "value": 0.1,
      "required": true,
      "description": " mnist learning_rate "
    },
    {
      "name": "training.batch_size",
      "value": 150,
      "required": false,
      "description": "This is batch size of training"
    }
  ],
  "experimentSpec": {
    "meta": {
      "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{training.learning_rate}} --batch_size={{training.batch_size}}",
      "name": "tf-mnist-template-test",
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
      "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0"
    }
  }
}
```

### Register experiment template
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "name": "tf-mnist-test",
  "author": "author",
  "description": "This is a template to run tf-mnist\n",
  "parameters": [
    {
      "name": "training.learning_rate",
      "value": 0.1,
      "required": true,
      "description": " mnist learning_rate "
    },
    {
      "name": "training.batch_size",
      "value": 150,
      "required": false,
      "description": "This is batch size of training"
    }
  ],
  "experimentSpec": {
    "meta": {
      "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{training.learning_rate}} --batch_size={{training.batch_size}}",
      "name": "tf-mnist-template-test",
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
      "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0"
    }
  }
}
' http://127.0.0.1:8080/api/v1/template
```

JSON Format example:
```json
{
    "name": "tf-mnist-test", 
    "params": {
        "training.learning_rate":"0.01", 
        "training.batch_size":"150"
    }
}
```

### Submit experiment template
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
    "name": "tf-mnist-test", 
    "params": {
        "training.learning_rate":"0.01", 
        "training.batch_size":"150"
    }
}
' http://127.0.0.1:8080/api/v1/template/submit
```

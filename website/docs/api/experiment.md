---
title: Experiment REST API
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

> Note: The Experiment API is in the alpha stage which is subjected to incompatible changes in
> future releases.

## Create Experiment (Using Anonymous/Embedded Environment)
`POST /api/v1/experiment`

**Example Request**
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
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
      "resources": "cpu=1,memory=512M"
    },
    "Worker": {
      "replicas": 1,
      "resources": "cpu=1,memory=512M"
    }
  }
}
' http://127.0.0.1:8080/api/v1/experiment
```

**Example Response:**
```json
{
  "status": "OK",
  "code": 200,
  "result": {
    "experimentId": "experiment_1586156073228_0001",
    "name": "tf-mnist-json",
    "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
    "status": "Accepted",
    "acceptedTime": "2020-06-13T22:59:29.000+08:00",
    "spec": {
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
          "resources": "cpu=1,memory=512M"
        },
        "Worker": {
          "replicas": 1,
          "resources": "cpu=1,memory=512M"
        }
      }
    }
  }
}
```

## Create Experiment (Using Pre-defined/Stored Environment)
`POST /api/v1/experiment`

**Example Request**
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
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
    "name": "my-submarine-env"
  },
  "spec": {
    "Ps": {
      "replicas": 1,
      "resources": "cpu=1,memory=512M"
    },
    "Worker": {
      "replicas": 1,
      "resources": "cpu=1,memory=512M"
    }
  }
}
' http://127.0.0.1:8080/api/v1/experiment
```
Above example assume environment "my-submarine-env" already exists in Submarine. Please refer Environment API Reference doc to Create/Update/Delete/List Environment REST API's

**Example Response:**
```json
{
  "status": "OK",
  "code": 200,
  "result": {
    "experimentId": "experiment_1586156073228_0001",
    "name": "tf-mnist-json",
    "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
    "status": "Accepted",
    "acceptedTime": "2020-06-13T22:59:29.000+08:00",
    "spec": {
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
        "name": "my-submarine-env"
      },
      "spec": {
        "Ps": {
          "replicas": 1,
          "resources": "cpu=1,memory=512M"
        },
        "Worker": {
          "replicas": 1,
          "resources": "cpu=1,memory=512M"
        }
      }
    }
  }
}
```

### List experiment
`GET /api/v1/experiment`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/experiment
```

**Example Response:**
```json
{
    "status": "OK",
    "code": 200,
    "result": [
        {
            "experimentId": "experiment_1592057447228_0001",
            "name": "tf-mnist-json",
            "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
            "status": "Accepted",
            "acceptedTime": "2020-06-13T22:59:29.000+08:00",
            "spec": {
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
                        "resources": "cpu=1,memory=512M"
                    },
                    "Worker": {
                        "replicas": 1,
                        "resources": "cpu=1,memory=512M"
                    }
                }
            }
        },
        {
            "experimentId": "experiment_1592057447228_0002",
            "name": "mnist",
            "uid": "38e39dcd-77d4-11ea-8dbb-0242ac110003",
            "status": "Accepted",
            "acceptedTime": "2020-06-13T22:19:29.000+08:00",
            "spec": {
                "meta": {
                    "name": "pytorch-mnist-json",
                    "namespace": "default",
                    "framework": "PyTorch",
                    "cmd": "python /var/mnist.py --backend gloo",
                    "envVars": {
                        "ENV_1": "ENV1"
                    }
                },
                "environment": {
                    "image": "apache/submarine:pytorch-dist-mnist-1.0"
                },
                "spec": {
                    "Master": {
                        "replicas": 1,
                        "resources": "cpu=1,memory=1024M"
                    },
                    "Worker": {
                        "replicas": 1,
                        "resources": "cpu=1,memory=1024M"
                    }
                }
            }
        }
    ]
}
```

### Get experiment
`GET /api/v1/experiment/{id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/experiment/experiment_1592057447228_0001
```

**Example Response:**
```json
{
    "status": "OK",
    "code": 200,
    "result": {
        "experimentId": "experiment_1592057447228_0001",
        "name": "tf-mnist-json",
        "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
        "status": "Accepted",
        "acceptedTime": "2020-06-13T22:59:29.000+08:00",
        "spec": {
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
                    "resources": "cpu=1,memory=512M"
                },
                "Worker": {
                    "replicas": 1,
                    "resources": "cpu=1,memory=512M"
                }
            }
        }
    }
}
```

### Patch experiment
`PATCH /api/v1/experiment/{id}`

**Example Request:**
```sh
curl -X PATCH -H "Content-Type: application/json" -d '
{
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
      "resources": "cpu=1,memory=512M"
    },
    "Worker": {
      "replicas": 2,
      "resources": "cpu=1,memory=512M"
    }
  }
}
' http://127.0.0.1:8080/api/v1/experiment/experiment_1592057447228_0001
```

**Example Response:**
```json
{
    "status": "OK",
    "code": 200,
    "success": true,
    "result": {
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
                "resources": "cpu=1,memory=512M"
            },
            "Worker": {
                "replicas": 2,
                "resources": "cpu=1,memory=512M"
            }
        }
    }
}
```

### Delete experiment
`GET /api/v1/experiment/{id}`

**Example Request:**
```sh
curl -X DELETE http://127.0.0.1:8080/api/v1/experiment/experiment_1592057447228_0001
```

**Example Response:**
```json
{
    "status": "OK",
    "code": 200,
    "result": {
        "experimentId": "experiment_1586156073228_0001",
        "name": "tf-mnist-json",
        "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
        "status": "Accepted",
        "acceptedTime": "2020-06-13T22:59:29.000+08:00",
        "spec": {
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
                    "resources": "cpu=1,memory=512M"
                },
                "Worker": {
                    "replicas": 2,
                    "resources": "cpu=1,memory=512M"
                }
            }
        }
    }
}
```

### List experiment Log
`GET /api/v1/experiment/logs`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/experiment/logs
```

**Example Response:**
```json
{
    "status": "OK",
    "code": 200,
    "success": null,
    "message": null,
    "result": [
        {
            "experimentId": "experiment_1589199154923_0001",
            "logContent": [
                {
                    "podName": "mnist-worker-0",
                    "podLog": null
                }
            ]
        },
        {
            "experimentId": "experiment_1589199154923_0002",
            "logContent": [
                {
                    "podName": "pytorch-dist-mnist-gloo-master-0",
                    "podLog": null
                },
                {
                    "podName": "pytorch-dist-mnist-gloo-worker-0",
                    "podLog": null
                }
            ]
        }
    ],
    "attributes": {}
}
```

### Get experiment Log
`GET /api/v1/experiment/logs/{id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/experiment/logs/experiment_1589199154923_0002
```

**Example Response:**
```json
{
    "status": "OK",
    "code": 200,
    "success": null,
    "message": null,
    "result": {
        "experimentId": "experiment_1589199154923_0002",
        "logContent": [
            {
                "podName": "pytorch-dist-mnist-gloo-master-0",
                "podLog": "Using distributed PyTorch with gloo backend\nDownloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\nDownloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\nDownloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\nDownloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\nProcessing...\nDone!\nTrain Epoch: 1 [0/60000 (0%)]\tloss=2.3000\nTrain Epoch: 1 [640/60000 (1%)]\tloss=2.2135\nTrain Epoch: 1 [1280/60000 (2%)]\tloss=2.1704\nTrain Epoch: 1 [1920/60000 (3%)]\tloss=2.0766\nTrain Epoch: 1 [2560/60000 (4%)]\tloss=1.8679\nTrain Epoch: 1 [3200/60000 (5%)]\tloss=1.4135\nTrain Epoch: 1 [3840/60000 (6%)]\tloss=1.0003\nTrain Epoch: 1 [4480/60000 (7%)]\tloss=0.7762\nTrain Epoch: 1 [5120/60000 (9%)]\tloss=0.4598\nTrain Epoch: 1 [5760/60000 (10%)]\tloss=0.4860\nTrain Epoch: 1 [6400/60000 (11%)]\tloss=0.4389\nTrain Epoch: 1 [7040/60000 (12%)]\tloss=0.4084\nTrain Epoch: 1 [7680/60000 (13%)]\tloss=0.4602\nTrain Epoch: 1 [8320/60000 (14%)]\tloss=0.4289\nTrain Epoch: 1 [8960/60000 (15%)]\tloss=0.3990\nTrain Epoch: 1 [9600/60000 (16%)]\tloss=0.3852\n"
            },
            {
                "podName": "pytorch-dist-mnist-gloo-worker-0",
                "podLog": "Using distributed PyTorch with gloo backend\nDownloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\nDownloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\nDownloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\nDownloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\nProcessing...\nDone!\nTrain Epoch: 1 [0/60000 (0%)]\tloss=2.3000\nTrain Epoch: 1 [640/60000 (1%)]\tloss=2.2135\nTrain Epoch: 1 [1280/60000 (2%)]\tloss=2.1704\nTrain Epoch: 1 [1920/60000 (3%)]\tloss=2.0766\nTrain Epoch: 1 [2560/60000 (4%)]\tloss=1.8679\nTrain Epoch: 1 [3200/60000 (5%)]\tloss=1.4135\nTrain Epoch: 1 [3840/60000 (6%)]\tloss=1.0003\nTrain Epoch: 1 [4480/60000 (7%)]\tloss=0.7762\nTrain Epoch: 1 [5120/60000 (9%)]\tloss=0.4598\nTrain Epoch: 1 [5760/60000 (10%)]\tloss=0.4860\nTrain Epoch: 1 [6400/60000 (11%)]\tloss=0.4389\nTrain Epoch: 1 [7040/60000 (12%)]\tloss=0.4084\nTrain Epoch: 1 [7680/60000 (13%)]\tloss=0.4602\nTrain Epoch: 1 [8320/60000 (14%)]\tloss=0.4289\nTrain Epoch: 1 [8960/60000 (15%)]\tloss=0.3990\nTrain Epoch: 1 [9600/60000 (16%)]\tloss=0.3852\n"
            }
        ]
    },
    "attributes": {}
}
```

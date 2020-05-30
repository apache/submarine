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

# Submarine Server Guide
This guide covers the deploy and running the training job by submarine server.
It now supports Tensorflow and PyTorch jobs.

## Prepare environment
- Java 1.8.x or higher.
- A K8s cluster
- The Docker image encapsulated with your deep learning application code

Note that We provide a learning and production environment tutorial. For more deployment info see [Deploy Submarine Server on Kubernetes](project/github/submarine/docs/userdocs/k8s/setup-kubernetes.md).

## Training
A generic job spec was designed for training job request, you should get familiar with the the job spec before submit job.

### Job Spec
Job spec consists of `librarySpec`, `submitterSpec` and `taskSpecs`. Below are examples of the spec:

### Sample Tensorflow Spec
```yaml
name: "mnist"
namespace: "submarine"
librarySpec:
  name: "TensorFlow"
  version: "2.1.0"
  image: "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0"
  cmd: "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150"
  envVars:
    ENV_1: "ENV1"
taskSpecs:
  Ps:
    name: tensorflow
    replicas: 2
    resources: "cpu=4,memory=2048M,nvidia.com/gpu=1"
  Worker:
    name: tensorflow
    replicas: 2
    resources: "cpu=4,memory=2048M,nvidia.com/gpu=1"
```
or
```json
{
  "name": "mnist",
  "namespace": "submarine",
  "librarySpec": {
    "name": "TensorFlow",
    "version": "2.1.0",
    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "taskSpecs": {
    "Ps": {
      "name": "tensorflow",
      "replicas": 2,
      "resources": "cpu=4,memory=2048M,nvidia.com/gpu=1"
    },
    "Worker": {
      "name": "tensorflow",
      "replicas": 2,
      "resources": "cpu=4,memory=2048M,nvidia.com/gpu=1"
    }
  }
}
```

### Sample PyTorch Spec

```json
{
  "name": "pytorch-dist-mnist-gloo",
  "namespace": "submarine",
  "librarySpec": {
    "name": "pytorch",
    "version": "2.1.0",
    "image": "apache/submarine:pytorch-dist-mnist-1.0",
    "cmd": "python /var/mnist.py --backend gloo",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "taskSpecs": {
    "Master": {
      "name": "master",
      "replicas": 1,
      "resources": "cpu=1,memory=1024M"
    },
    "Worker": {
      "name": "worker",
      "replicas": 1,
      "resources": "cpu=1,memory=1024M"
    }
  }
}
```

For more info about the spec definition see [here](project/github/submarine/docs/design/submarine-server/jobspec.md).

## Job Operation by REST API
### Create Job
`POST /api/v1/jobs`

**Example Request**
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "name": "mnist",
  "namespace": "submarine",
  "librarySpec": {
    "name": "TensorFlow",
    "version": "2.1.0",
    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "taskSpecs": {
    "Worker": {
      "name": "tensorflow",
      "replicas": 1,
      "resources": "cpu=1,memory=1024M"
    }
  }
}
' http://127.0.0.1/api/v1/jobs
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "result": {
        "jobId": "job_1586156073228_0005",
        "name": "mnist",
        "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
        "status": "Accepted",
        "acceptedTime": "2020-04-06T14:59:29.000+08:00",
        "spec": {
            "name": "mnist",
            "namespace": "submarine",
            "librarySpec": {
                "name": "TensorFlow",
                "version": "2.1.0",
                "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                "envVars": {
                    "ENV_1": "ENV1"
                }
            },
            "taskSpecs": {
                "Worker": {
                    "name": "tensorflow",
                    "resources": "cpu=1,memory=1024M",
                    "replicas": 1,
                    "resourceMap": {
                        "memory": "1024M",
                        "cpu": "1"
                    }
                }
            }
        }
    }
}
```

### List Jobs
`GET /api/v1/jobs`

**Example Request:**
```sh
curl -X GET http://127.0.0.1/api/v1/jobs
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "result": [
        {
            "jobId": "job_1586156073228_0005",
            "name": "mnist",
            "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
            "status": "Created",
            "acceptedTime": "2020-04-06T14:59:29.000+08:00",
            "createdTime": "2020-04-06T14:59:29.000+08:00",
            "spec": {
                "name": "mnist",
                "namespace": "submarine",
                "librarySpec": {
                    "name": "TensorFlow",
                    "version": "2.1.0",
                    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                    "envVars": {
                        "ENV_1": "ENV1"
                    }
                },
                "taskSpecs": {
                    "Worker": {
                        "name": "tensorflow",
                        "resources": "cpu=1,memory=1024M",
                        "replicas": 1,
                        "resourceMap": {
                            "memory": "1024M",
                            "cpu": "1"
                        }
                    }
                }
            }
        }
    ]
}
```

### Get Job
`GET /api/v1/jobs/{id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1/api/v1/jobs/job_1586156073228_0005
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "result": {
        "jobId": "job_1586156073228_0005",
        "name": "mnist",
        "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
        "status": "Created",
        "acceptedTime": "2020-04-06T14:59:29.000+08:00",
        "createdTime": "2020-04-06T14:59:29.000+08:00",
        "spec": {
            "name": "mnist",
            "namespace": "submarine",
            "librarySpec": {
                "name": "TensorFlow",
                "version": "2.1.0",
                "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                "envVars": {
                    "ENV_1": "ENV1"
                }
            },
            "taskSpecs": {
                "Worker": {
                    "name": "tensorflow",
                    "resources": "cpu=1,memory=1024M",
                    "replicas": 1,
                    "resourceMap": {
                        "memory": "1024M",
                        "cpu": "1"
                    }
                }
            }
        }
    }
}
```

### Patch Job
`PATCH /api/v1/jobs/{id}`

**Example Request:**
```sh
curl -X PATCH -H "Content-Type: application/json" -d '
{
  "name": "mnist",
  "namespace": "submarine",
  "librarySpec": {
    "name": "TensorFlow",
    "version": "2.1.0",
    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "taskSpecs": {
    "Worker": {
      "name": "tensorflow",
      "replicas": 2,
      "resources": "cpu=1,memory=1024M"
    }
  }
}
' http://127.0.0.1/api/v1/jobs/job_1586156073228_0005
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "success": true,
    "result": {
        "jobId": "job_1586156073228_0005",
        "name": "mnist",
        "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
        "status": "Created",
        "acceptedTime": "2020-04-06T14:59:29.000+08:00",
        "createdTime": "2020-04-06T14:59:29.000+08:00",
        "spec": {
            "name": "mnist",
            "namespace": "submarine",
            "librarySpec": {
                "name": "TensorFlow",
                "version": "2.1.0",
                "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                "envVars": {
                    "ENV_1": "ENV1"
                }
            },
            "taskSpecs": {
                "Worker": {
                    "name": "tensorflow",
                    "resources": "cpu=1,memory=1024M",
                    "replicas": 2,
                    "resourceMap": {
                        "memory": "1024M",
                        "cpu": "1"
                    }
                }
            }
        }
    }
}
```

### Delete Job
`GET /api/v1/jobs/{id}`

**Example Request:**
```sh
curl -X DELETE http://127.0.0.1/api/v1/jobs/job_123_01
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "result": {
        "jobId": "job_1586156073228_0005",
        "name": "mnist",
        "uid": "28e39dcd-77d4-11ea-8dbb-0242ac110003",
        "status": "Deleted",
        "acceptedTime": "2020-04-06T14:59:29.000+08:00",
        "createdTime": "2020-04-06T14:59:29.000+08:00",
        "spec": {
            "name": "mnist",
            "namespace": "submarine",
            "librarySpec": {
                "name": "TensorFlow",
                "version": "2.1.0",
                "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                "envVars": {
                    "ENV_1": "ENV1"
                }
            },
            "taskSpecs": {
                "Worker": {
                    "name": "tensorflow",
                    "resources": "cpu=1,memory=1024M",
                    "replicas": 1,
                    "resourceMap": {
                        "memory": "1024M",
                        "cpu": "1"
                    }
                }
            }
        }
    }
}
```

### List Job Log
`GET /api/v1/jobs/logs`

**Example Request:**
```sh
curl -X GET http://127.0.0.1/api/v1/jobs/logs
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "success": null,
    "message": null,
    "result": [
        {
            "jobId": "job_1589199154923_0001",
            "logContent": [
                {
                    "podName": "mnist-worker-0",
                    "podLog": null
                }
            ]
        },
        {
            "jobId": "job_1589199154923_0002",
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

### Get Job Log
`GET /api/v1/jobs/logs/{id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1/api/v1/jobs/logs/job_1589199154923_0002
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "success": null,
    "message": null,
    "result": {
        "jobId": "job_1589199154923_0002",
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

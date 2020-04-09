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

Note that We provide a learning and production environment tutorial. For more deployment info see [Deploy Submarine Server on Kubernetes](./setup-kubernetes.md).

## Training
A generic job spec was designed for training job request, you should get familiar with the the job spec before submit job.

### Job Spec
Job spec consists of `librarySpec`, `submitterSpec` and `taskSpecs`. Below are examples of the spec:

### Sample Tensorflow Spec
```yaml
name: "mnist"
librarySpec:
  name: "TensorFlow"
  version: "2.1.0"
  image: "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0"
  cmd: "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150"
  envVars:
    ENV_1: "ENV1"
submitterSpec:
  type: "k8s"
  configPath:
  namespace: "submarine"
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
  "librarySpec": {
    "name": "TensorFlow",
    "version": "2.1.0",
    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "submitterSpec": {
    "type": "k8s",
    "namespace": "submarine"
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
  "librarySpec": {
    "name": "pytorch",
    "version": "2.1.0",
    "image": "apache/submarine:pytorch-dist-mnist-1.0",
    "cmd": "python /var/mnist.py --backend gloo",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "submitterSpec": {
    "type": "k8s",
    "namespace": "submarine"
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

For more info about the spec definition see [here](../design/submarine-server/jobspec.md).

## Job Operation by REST API
### Create Job
`POST /api/v1/jobs`

**Example Request**
```sh
curl -X POST -H "Content-Type: application/json" -d '
{
  "name": "mnist",
  "librarySpec": {
    "name": "TensorFlow",
    "version": "2.1.0",
    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "submitterSpec": {
    "type": "k8s",
    "namespace": "submarine"
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
            "librarySpec": {
                "name": "TensorFlow",
                "version": "2.1.0",
                "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                "envVars": {
                    "ENV_1": "ENV1"
                }
            },
            "submitterSpec": {
                "type": "k8s",
                "namespace": "submarine"
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
                "librarySpec": {
                    "name": "TensorFlow",
                    "version": "2.1.0",
                    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                    "envVars": {
                        "ENV_1": "ENV1"
                    }
                },
                "submitterSpec": {
                    "type": "k8s",
                    "namespace": "submarine"
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
            "librarySpec": {
                "name": "TensorFlow",
                "version": "2.1.0",
                "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                "envVars": {
                    "ENV_1": "ENV1"
                }
            },
            "submitterSpec": {
                "type": "k8s",
                "namespace": "submarine"
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
  "librarySpec": {
    "name": "TensorFlow",
    "version": "2.1.0",
    "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
    "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "submitterSpec": {
    "type": "k8s",
    "namespace": "submarine"
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
            "librarySpec": {
                "name": "TensorFlow",
                "version": "2.1.0",
                "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                "envVars": {
                    "ENV_1": "ENV1"
                }
            },
            "submitterSpec": {
                "type": "k8s",
                "namespace": "submarine"
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
            "librarySpec": {
                "name": "TensorFlow",
                "version": "2.1.0",
                "image": "gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0",
                "cmd": "python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate=0.01 --batch_size=150",
                "envVars": {
                    "ENV_1": "ENV1"
                }
            },
            "submitterSpec": {
                "type": "k8s",
                "namespace": "submarine"
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

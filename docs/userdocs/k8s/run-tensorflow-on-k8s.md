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

# Running TensorFlow job on K8s
This guide covers the running the training **TensorFlow** job by submarine server.
It now supports TensorFlow and PyTorch jobs.

## Prepare environment
- Java 1.8.x or higher.
- A K8s cluster
- The Docker image encapsulated with your deep learning application code

Note that We provide a learning and production environment tutorial. For more deployment info see [Deploy Submarine Server on Kubernetes](./setup-kubernetes.md).

## Job Spec
A generic job spec was designed for training job request, you should get familiar with the the job spec before submit job.

For more info about the spec definition see [here](../../design/submarine-server/jobspec.md).

Job spec consists of `librarySpec`, `submitterSpec` and `taskSpecs`. Below are examples of the spec:

### Sample TensorFlow Spec
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
' http://127.0.0.1:8080/api/v1/jobs
```

**Example Response:**
```sh
{
  "status": "OK",
  "code": 200,
  "success": true,
  "message": null,
  "result": {
    "jobId": "job_1591601852376_0004",
    "name": "mnist444",
    "uid": "d28ce6b7-d781-4cad-bce4-5b82cc6853cd",
    "status": "Accepted",
    "acceptedTime": "2020-06-12T22:30:11.000+08:00",
    "createdTime": null,
    "runningTime": null,
    "finishedTime": null,
    "spec": {
      "name": "mnist444",
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
          "image": null,
          "cmd": null,
          "envVars": null,
          "resources": "cpu=1,memory=1024M",
          "replicas": 1,
          "resourceMap": {
            "memory": "1024M",
            "cpu": "1"
          }
        }
      },
      "projects": null
    }
  },
  "attributes": {}
}
```

### List Jobs
`GET /api/v1/jobs`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/jobs
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "success": true,
    "message": null,
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
curl -X GET http://127.0.0.1:8080/api/v1/jobs/job_1586156073228_0005
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "success": true,
    "message": null,
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
    "message": null,
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
curl -X DELETE http://127.0.0.1:8080/api/v1/jobs/job_1586156073228_0005
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "success": true,
    "message": null,
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
curl -X GET http://127.0.0.1:8080/api/v1/jobs/logs
```

**Example Response:**
```sh
{
    "status": "OK",
    "code": 200,
    "success": true,
    "message": null,
    "result": [
        {
            "jobId": "job_1586156073228_0005",
            "logContent": [
                {
                    "podName": "mnist-worker-0",
                    "podLog": []
                }
            ]
        },
    "attributes": {}
}
```

### Get Job Log
`GET /api/v1/jobs/logs/{id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:8080/api/v1/jobs/logs/job_1586156073228_0005
```

**Example Response:**
```sh
{
  "status": "OK",
  "code": 200,
  "success": true,
  "message": null,
  "result": {
    "jobId": "job_1586156073228_0005",
    "logContent": [
      {
        "podName": "mnist-worker-0",
        "podLog": [
          "WARNING:tensorflow:From /var/tf_mnist/mnist_with_summaries.py:39: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.",
          "Instructions for updating:",
          "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.",
          "WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.",
          "Instructions for updating:",
          "Please write your own downloading logic.",
          "WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:252: wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.",
          "Instructions for updating:",
          "Please use urllib or similar directly.",
          "WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.",
          "Instructions for updating:",
          "Please use tf.data to implement this functionality.",
          "WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.",
          "Instructions for updating:",
          "Please use tf.data to implement this functionality.",
          "WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: __init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.",
          "Instructions for updating:",
          "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.",
          "2020-06-08 15:11:12.307831: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA",
          "Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz",
          "Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz",
          "Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz",
          "Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz",
          "Accuracy at step 0: 0.0634",
          "Accuracy at step 10: 0.7151",
          "Accuracy at step 20: 0.8699",
          "Accuracy at step 30: 0.9028",
          "Accuracy at step 40: 0.9189",
          "Accuracy at step 50: 0.9205",
          "Accuracy at step 60: 0.9231",
          "Accuracy at step 70: 0.9362",
          "Accuracy at step 80: 0.9403",
          "Accuracy at step 90: 0.9387",
          "Adding run metadata for 99",
          "Accuracy at step 100: 0.9443",
          "Accuracy at step 110: 0.9432",
          "Accuracy at step 120: 0.9459",
          "Accuracy at step 130: 0.9466",
          "Accuracy at step 140: 0.9451",
          "Accuracy at step 150: 0.9473",
          "Accuracy at step 160: 0.9522",
          "Accuracy at step 170: 0.9484",
          "Accuracy at step 180: 0.954",
          "Accuracy at step 190: 0.9491",
          "Adding run metadata for 199",
          "Accuracy at step 200: 0.9541",
          "Accuracy at step 210: 0.9526",
          "Accuracy at step 220: 0.9561",
          "Accuracy at step 230: 0.9526",
          "Accuracy at step 240: 0.9543",
          "Accuracy at step 250: 0.9593",
          "Accuracy at step 260: 0.9548",
          "Accuracy at step 270: 0.9566",
          "Accuracy at step 280: 0.9568",
          "Accuracy at step 290: 0.9552",
          "Adding run metadata for 299",
          "Accuracy at step 300: 0.9541",
          "Accuracy at step 310: 0.962",
          "Accuracy at step 320: 0.9599",
          "Accuracy at step 330: 0.9626",
          "Accuracy at step 340: 0.9611",
          "Accuracy at step 350: 0.9621",
          "Accuracy at step 360: 0.9569",
          "Accuracy at step 370: 0.9585",
          "Accuracy at step 380: 0.9632",
          "Accuracy at step 390: 0.9549",
          "Adding run metadata for 399",
          "Accuracy at step 400: 0.9613",
          "Accuracy at step 410: 0.9605",
          "Accuracy at step 420: 0.9616",
          "Accuracy at step 430: 0.9618",
          "Accuracy at step 440: 0.9603",
          "Accuracy at step 450: 0.9642",
          "Accuracy at step 460: 0.9657",
          "Accuracy at step 470: 0.9633",
          "Accuracy at step 480: 0.9653",
          "Accuracy at step 490: 0.962",
          "Adding run metadata for 499",
          "Accuracy at step 500: 0.9674",
          "Accuracy at step 510: 0.966",
          "Accuracy at step 520: 0.9651",
          "Accuracy at step 530: 0.9611",
          "Accuracy at step 540: 0.9655",
          "Accuracy at step 550: 0.9655",
          "Accuracy at step 560: 0.9641",
          "Accuracy at step 570: 0.9666",
          "Accuracy at step 580: 0.9623",
          "Accuracy at step 590: 0.9622",
          "Adding run metadata for 599",
          "Accuracy at step 600: 0.9632",
          "Accuracy at step 610: 0.965",
          "Accuracy at step 620: 0.9639",
          "Accuracy at step 630: 0.9649",
          "Accuracy at step 640: 0.9669",
          "Accuracy at step 650: 0.9691",
          "Accuracy at step 660: 0.9666",
          "Accuracy at step 670: 0.9683",
          "Accuracy at step 680: 0.9692",
          "Accuracy at step 690: 0.9671",
          "Adding run metadata for 699",
          "Accuracy at step 700: 0.9635",
          "Accuracy at step 710: 0.9635",
          "Accuracy at step 720: 0.9643",
          "Accuracy at step 730: 0.9585",
          "Accuracy at step 740: 0.9606",
          "Accuracy at step 750: 0.9644",
          "Accuracy at step 760: 0.9635",
          "Accuracy at step 770: 0.9656",
          "Accuracy at step 780: 0.9639",
          "Accuracy at step 790: 0.9607",
          "Adding run metadata for 799",
          "Accuracy at step 800: 0.9593",
          "Accuracy at step 810: 0.9595",
          "Accuracy at step 820: 0.9636",
          "Accuracy at step 830: 0.9632",
          "Accuracy at step 840: 0.9695",
          "Accuracy at step 850: 0.9682",
          "Accuracy at step 860: 0.966",
          "Accuracy at step 870: 0.9673",
          "Accuracy at step 880: 0.9696",
          "Accuracy at step 890: 0.9707",
          "Adding run metadata for 899",
          "Accuracy at step 900: 0.9659",
          "Accuracy at step 910: 0.9647",
          "Accuracy at step 920: 0.9666",
          "Accuracy at step 930: 0.9702",
          "Accuracy at step 940: 0.9664",
          "Accuracy at step 950: 0.9624",
          "Accuracy at step 960: 0.9608",
          "Accuracy at step 970: 0.9641",
          "Accuracy at step 980: 0.9649",
          "Accuracy at step 990: 0.963",
          "Adding run metadata for 999"
        ]
      }
    ]
  },
  "attributes": {}
}
```

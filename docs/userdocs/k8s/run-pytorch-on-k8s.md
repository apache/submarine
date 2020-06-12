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

# Running PyTorch job on K8s
This guide covers the running the training **PyTorch** job by submarine server.
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
## Job Operation by REST API
### Create Job
`POST /api/v1/jobs`

**Example Request**
```sh
curl -X POST -H "Content-Type: application/json" -d '
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
    "jobId": "job_1591601852376_0005",
    "name": "pytorch-dist-mnist-gloo",
    "uid": "0df525cf-46d9-4925-9ba5-6c02800478ad",
    "status": "Accepted",
    "acceptedTime": "2020-06-12T22:48:36.000+08:00",
    "createdTime": null,
    "runningTime": null,
    "finishedTime": null,
    "spec": {
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
          "image": null,
          "cmd": null,
          "envVars": null,
          "resources": "cpu=1,memory=1024M",
          "replicas": 1,
          "resourceMap": {
            "memory": "1024M",
            "cpu": "1"
          }
        },
        "Worker": {
          "name": "worker",
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
      "jobId": "job_1591601852376_0005",
      "name": "pytorch-dist-mnist-gloo",
      "uid": "0df525cf-46d9-4925-9ba5-6c02800478ad",
      "status": "Running",
      "acceptedTime": "2020-06-12T22:48:36.000+08:00",
      "createdTime": "2020-06-12T22:48:36.000+08:00",
      "runningTime": "2020-06-12T22:49:52.000+08:00",
      "finishedTime": null,
      "spec": {
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
            "image": null,
            "cmd": null,
            "envVars": null,
            "resources": "cpu=1,memory=1024M",
            "replicas": 1,
            "resourceMap": {
              "memory": "1024M",
              "cpu": "1"
            }
          },
          "Worker": {
            "name": "worker",
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
    }
  ],
  "attributes": {}
}
```

### Get Job
`GET /api/v1/jobs/{id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1/api/v1/jobs/job_1591601852376_0005
```

**Example Response:**
```sh
{
  "status": "OK",
  "code": 200,
  "success": true,
  "message": null,
  "result": {
    "jobId": "job_1591601852376_0005",
    "name": "pytorch-dist-mnist-gloo",
    "uid": "0df525cf-46d9-4925-9ba5-6c02800478ad",
    "status": "Running",
    "acceptedTime": "2020-06-12T22:48:36.000+08:00",
    "createdTime": "2020-06-12T22:48:36.000+08:00",
    "runningTime": "2020-06-12T22:49:52.000+08:00",
    "finishedTime": null,
    "spec": {
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
          "image": null,
          "cmd": null,
          "envVars": null,
          "resources": "cpu=1,memory=1024M",
          "replicas": 1,
          "resourceMap": {
            "memory": "1024M",
            "cpu": "1"
          }
        },
        "Worker": {
          "name": "worker",
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

### Patch Job
`PATCH /api/v1/jobs/{id}`

**Example Request:**
```sh
curl -X PATCH -H "Content-Type: application/json" -d '
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
      "resources": "cpu=2,memory=2048M"
    },
    "Worker": {
      "name": "worker",
      "replicas": 1,
      "resources": "cpu=2,memory=2048"
    }
  }
}
' http://127.0.0.1:8080/api/v1/jobs/job_1591601852376_0005
```

**Example Response:**
```sh
{
  "status": "OK",
  "code": 200,
  "success": true,
  "message": null,
  "result": {
    "jobId": "job_1591601852376_0005",
    "name": "pytorch-dist-mnist-gloo",
    "uid": "0df525cf-46d9-4925-9ba5-6c02800478ad",
    "status": "Running",
    "acceptedTime": "2020-06-12T22:48:36.000+08:00",
    "createdTime": "2020-06-12T22:48:36.000+08:00",
    "runningTime": "2020-06-12T22:49:52.000+08:00",
    "finishedTime": null,
    "spec": {
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
          "image": null,
          "cmd": null,
          "envVars": null,
          "resources": "cpu=2,memory=2048M",
          "replicas": 1,
          "resourceMap": {
            "memory": "2048M",
            "cpu": "2"
          }
        },
        "Worker": {
          "name": "worker",
          "image": null,
          "cmd": null,
          "envVars": null,
          "resources": "cpu=2,memory=2048",
          "replicas": 1,
          "resourceMap": {
            "memory": "2048",
            "cpu": "2"
          }
        }
      },
      "projects": null
    }
  },
  "attributes": {}
}
```

### Delete Job
`GET /api/v1/jobs/{id}`
**Example Request:**
```sh
curl -X DELETE http://127.0.0.1:8080/api/v1/jobs/job_1591601852376_0005
```

**Example Response:**
```sh
{
  "status": "OK",
  "code": 200,
  "success": true,
  "message": null,
  "result": {
    "jobId": "job_1591601852376_0005",
    "name": "pytorch-dist-mnist-gloo",
    "uid": "0df525cf-46d9-4925-9ba5-6c02800478ad",
    "status": "Deleted",
    "acceptedTime": "2020-06-12T22:48:36.000+08:00",
    "createdTime": "2020-06-12T22:48:36.000+08:00",
    "runningTime": "2020-06-12T22:49:52.000+08:00",
    "finishedTime": "2020-06-13T01:00:53.000+08:00",
    "spec": {
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
          "image": null,
          "cmd": null,
          "envVars": null,
          "resources": "cpu=2,memory=2048M",
          "replicas": 1,
          "resourceMap": {
            "memory": "2048M",
            "cpu": "2"
          }
        },
        "Worker": {
          "name": "worker",
          "image": null,
          "cmd": null,
          "envVars": null,
          "resources": "cpu=2,memory=2048",
          "replicas": 1,
          "resourceMap": {
            "memory": "2048",
            "cpu": "2"
          }
        }
      },
      "projects": null
    }
  },
  "attributes": {}
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
      "jobId": "job_1591601852376_0005",
      "logContent": [
        {
          "podName": "pytorch-dist-mnist-gloo-master-0",
          "podLog": []
        },
        {
          "podName": "pytorch-dist-mnist-gloo-worker-0",
          "podLog": []
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
curl -X GET http://127.0.0.1:8080/api/v1/jobs/logs/job_1591601852376_0005
```

**Example Response:**
```sh
{
  "status": "OK",
  "code": 200,
  "success": true,
  "message": null,
  "result": {
    "jobId": "job_1591601852376_0005",
    "logContent": [
      {
        "podName": "pytorch-dist-mnist-gloo-master-0",
        "podLog": [
          "Using distributed PyTorch with gloo backend",
          "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
          "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
          "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
          "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
          "Processing...",
          "Done!",
          "Train Epoch: 1 [0/60000 (0%)]\tloss=2.3000",
          "Train Epoch: 1 [640/60000 (1%)]\tloss=2.2135",
          "Train Epoch: 1 [1280/60000 (2%)]\tloss=2.1704",
          "Train Epoch: 1 [1920/60000 (3%)]\tloss=2.0766",
          "Train Epoch: 1 [2560/60000 (4%)]\tloss=1.8679",
          "Train Epoch: 1 [3200/60000 (5%)]\tloss=1.4135",
          "Train Epoch: 1 [3840/60000 (6%)]\tloss=1.0003",
          "Train Epoch: 1 [4480/60000 (7%)]\tloss=0.7763",
          "Train Epoch: 1 [5120/60000 (9%)]\tloss=0.4598",
          "Train Epoch: 1 [5760/60000 (10%)]\tloss=0.4870",
          "Train Epoch: 1 [6400/60000 (11%)]\tloss=0.4381",
          "Train Epoch: 1 [7040/60000 (12%)]\tloss=0.4089",
          "Train Epoch: 1 [7680/60000 (13%)]\tloss=0.4618",
          "Train Epoch: 1 [8320/60000 (14%)]\tloss=0.4284",
          "Train Epoch: 1 [8960/60000 (15%)]\tloss=0.3992",
          "Train Epoch: 1 [9600/60000 (16%)]\tloss=0.3840",
          "Train Epoch: 1 [10240/60000 (17%)]\tloss=0.2981",
          "Train Epoch: 1 [10880/60000 (18%)]\tloss=0.5013",
          "Train Epoch: 1 [11520/60000 (19%)]\tloss=0.5246",
          "Train Epoch: 1 [12160/60000 (20%)]\tloss=0.3376",
          "Train Epoch: 1 [12800/60000 (21%)]\tloss=0.3678",
          "Train Epoch: 1 [13440/60000 (22%)]\tloss=0.4515",
          "Train Epoch: 1 [14080/60000 (23%)]\tloss=0.3043",
          "Train Epoch: 1 [14720/60000 (25%)]\tloss=0.3581",
          "Train Epoch: 1 [15360/60000 (26%)]\tloss=0.3301",
          "Train Epoch: 1 [16000/60000 (27%)]\tloss=0.4392",
          "Train Epoch: 1 [16640/60000 (28%)]\tloss=0.3626",
          "Train Epoch: 1 [17280/60000 (29%)]\tloss=0.3179",
          "Train Epoch: 1 [17920/60000 (30%)]\tloss=0.2013",
          "Train Epoch: 1 [18560/60000 (31%)]\tloss=0.5004",
          "Train Epoch: 1 [19200/60000 (32%)]\tloss=0.3266",
          "Train Epoch: 1 [19840/60000 (33%)]\tloss=0.1194",
          "Train Epoch: 1 [20480/60000 (34%)]\tloss=0.1898",
          "Train Epoch: 1 [21120/60000 (35%)]\tloss=0.1402",
          "Train Epoch: 1 [21760/60000 (36%)]\tloss=0.3160",
          "Train Epoch: 1 [22400/60000 (37%)]\tloss=0.1499",
          "Train Epoch: 1 [23040/60000 (38%)]\tloss=0.2887",
          "Train Epoch: 1 [23680/60000 (39%)]\tloss=0.4681",
          "Train Epoch: 1 [24320/60000 (41%)]\tloss=0.2160",
          "Train Epoch: 1 [24960/60000 (42%)]\tloss=0.1523",
          "Train Epoch: 1 [25600/60000 (43%)]\tloss=0.2243",
          "Train Epoch: 1 [26240/60000 (44%)]\tloss=0.2627",
          "Train Epoch: 1 [26880/60000 (45%)]\tloss=0.2337",
          "Train Epoch: 1 [27520/60000 (46%)]\tloss=0.2624",
          "Train Epoch: 1 [28160/60000 (47%)]\tloss=0.2127",
          "Train Epoch: 1 [28800/60000 (48%)]\tloss=0.1330",
          "Train Epoch: 1 [29440/60000 (49%)]\tloss=0.2779",
          "Train Epoch: 1 [30080/60000 (50%)]\tloss=0.0940",
          "Train Epoch: 1 [30720/60000 (51%)]\tloss=0.1270",
          "Train Epoch: 1 [31360/60000 (52%)]\tloss=0.2479",
          "Train Epoch: 1 [32000/60000 (53%)]\tloss=0.3388",
          "Train Epoch: 1 [32640/60000 (54%)]\tloss=0.1527",
          "Train Epoch: 1 [33280/60000 (55%)]\tloss=0.0908",
          "Train Epoch: 1 [33920/60000 (57%)]\tloss=0.1445",
          "Train Epoch: 1 [34560/60000 (58%)]\tloss=0.1979",
          "Train Epoch: 1 [35200/60000 (59%)]\tloss=0.2186",
          "Train Epoch: 1 [35840/60000 (60%)]\tloss=0.0632",
          "Train Epoch: 1 [36480/60000 (61%)]\tloss=0.1354",
          "Train Epoch: 1 [37120/60000 (62%)]\tloss=0.1163",
          "Train Epoch: 1 [37760/60000 (63%)]\tloss=0.2355",
          "Train Epoch: 1 [38400/60000 (64%)]\tloss=0.0634",
          "Train Epoch: 1 [39040/60000 (65%)]\tloss=0.1070",
          "Train Epoch: 1 [39680/60000 (66%)]\tloss=0.1599",
          "Train Epoch: 1 [40320/60000 (67%)]\tloss=0.1090",
          "Train Epoch: 1 [40960/60000 (68%)]\tloss=0.1770",
          "Train Epoch: 1 [41600/60000 (69%)]\tloss=0.2301",
          "Train Epoch: 1 [42240/60000 (70%)]\tloss=0.0745",
          "Train Epoch: 1 [42880/60000 (71%)]\tloss=0.1553",
          "Train Epoch: 1 [43520/60000 (72%)]\tloss=0.2793",
          "Train Epoch: 1 [44160/60000 (74%)]\tloss=0.1425",
          "Train Epoch: 1 [44800/60000 (75%)]\tloss=0.1168",
          "Train Epoch: 1 [45440/60000 (76%)]\tloss=0.1225",
          "Train Epoch: 1 [46080/60000 (77%)]\tloss=0.0774",
          "Train Epoch: 1 [46720/60000 (78%)]\tloss=0.1942",
          "Train Epoch: 1 [47360/60000 (79%)]\tloss=0.0686",
          "Train Epoch: 1 [48000/60000 (80%)]\tloss=0.2082",
          "Train Epoch: 1 [48640/60000 (81%)]\tloss=0.1393",
          "Train Epoch: 1 [49280/60000 (82%)]\tloss=0.0940",
          "Train Epoch: 1 [49920/60000 (83%)]\tloss=0.1073",
          "Train Epoch: 1 [50560/60000 (84%)]\tloss=0.1196",
          "Train Epoch: 1 [51200/60000 (85%)]\tloss=0.1445",
          "Train Epoch: 1 [51840/60000 (86%)]\tloss=0.0665",
          "Train Epoch: 1 [52480/60000 (87%)]\tloss=0.0242",
          "Train Epoch: 1 [53120/60000 (88%)]\tloss=0.2633",
          "Train Epoch: 1 [53760/60000 (90%)]\tloss=0.0916",
          "Train Epoch: 1 [54400/60000 (91%)]\tloss=0.1292",
          "Train Epoch: 1 [55040/60000 (92%)]\tloss=0.1909",
          "Train Epoch: 1 [55680/60000 (93%)]\tloss=0.0346",
          "Train Epoch: 1 [56320/60000 (94%)]\tloss=0.0358",
          "Train Epoch: 1 [56960/60000 (95%)]\tloss=0.0767",
          "Train Epoch: 1 [57600/60000 (96%)]\tloss=0.1175",
          "Train Epoch: 1 [58240/60000 (97%)]\tloss=0.1929",
          "Train Epoch: 1 [58880/60000 (98%)]\tloss=0.2051",
          "Train Epoch: 1 [59520/60000 (99%)]\tloss=0.0631",
          "",
          "accuracy=0.9668"
        ]
      },
      {
        "podName": "pytorch-dist-mnist-gloo-worker-0",
        "podLog": [
          "Using distributed PyTorch with gloo backend",
          "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
          "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
          "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
          "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
          "Processing...",
          "Done!",
          "Train Epoch: 1 [0/60000 (0%)]\tloss=2.3000",
          "Train Epoch: 1 [640/60000 (1%)]\tloss=2.2135",
          "Train Epoch: 1 [1280/60000 (2%)]\tloss=2.1704",
          "Train Epoch: 1 [1920/60000 (3%)]\tloss=2.0766",
          "Train Epoch: 1 [2560/60000 (4%)]\tloss=1.8679",
          "Train Epoch: 1 [3200/60000 (5%)]\tloss=1.4135",
          "Train Epoch: 1 [3840/60000 (6%)]\tloss=1.0003",
          "Train Epoch: 1 [4480/60000 (7%)]\tloss=0.7763",
          "Train Epoch: 1 [5120/60000 (9%)]\tloss=0.4598",
          "Train Epoch: 1 [5760/60000 (10%)]\tloss=0.4870",
          "Train Epoch: 1 [6400/60000 (11%)]\tloss=0.4381",
          "Train Epoch: 1 [7040/60000 (12%)]\tloss=0.4089",
          "Train Epoch: 1 [7680/60000 (13%)]\tloss=0.4618",
          "Train Epoch: 1 [8320/60000 (14%)]\tloss=0.4284",
          "Train Epoch: 1 [8960/60000 (15%)]\tloss=0.3992",
          "Train Epoch: 1 [9600/60000 (16%)]\tloss=0.3840",
          "Train Epoch: 1 [10240/60000 (17%)]\tloss=0.2981",
          "Train Epoch: 1 [10880/60000 (18%)]\tloss=0.5013",
          "Train Epoch: 1 [11520/60000 (19%)]\tloss=0.5246",
          "Train Epoch: 1 [12160/60000 (20%)]\tloss=0.3376",
          "Train Epoch: 1 [12800/60000 (21%)]\tloss=0.3678",
          "Train Epoch: 1 [13440/60000 (22%)]\tloss=0.4515",
          "Train Epoch: 1 [14080/60000 (23%)]\tloss=0.3043",
          "Train Epoch: 1 [14720/60000 (25%)]\tloss=0.3581",
          "Train Epoch: 1 [15360/60000 (26%)]\tloss=0.3301",
          "Train Epoch: 1 [16000/60000 (27%)]\tloss=0.4392",
          "Train Epoch: 1 [16640/60000 (28%)]\tloss=0.3626",
          "Train Epoch: 1 [17280/60000 (29%)]\tloss=0.3179",
          "Train Epoch: 1 [17920/60000 (30%)]\tloss=0.2013",
          "Train Epoch: 1 [18560/60000 (31%)]\tloss=0.5004",
          "Train Epoch: 1 [19200/60000 (32%)]\tloss=0.3266",
          "Train Epoch: 1 [19840/60000 (33%)]\tloss=0.1194",
          "Train Epoch: 1 [20480/60000 (34%)]\tloss=0.1898",
          "Train Epoch: 1 [21120/60000 (35%)]\tloss=0.1402",
          "Train Epoch: 1 [21760/60000 (36%)]\tloss=0.3160",
          "Train Epoch: 1 [22400/60000 (37%)]\tloss=0.1499",
          "Train Epoch: 1 [23040/60000 (38%)]\tloss=0.2887",
          "Train Epoch: 1 [23680/60000 (39%)]\tloss=0.4681",
          "Train Epoch: 1 [24320/60000 (41%)]\tloss=0.2160",
          "Train Epoch: 1 [24960/60000 (42%)]\tloss=0.1523",
          "Train Epoch: 1 [25600/60000 (43%)]\tloss=0.2243",
          "Train Epoch: 1 [26240/60000 (44%)]\tloss=0.2627",
          "Train Epoch: 1 [26880/60000 (45%)]\tloss=0.2337",
          "Train Epoch: 1 [27520/60000 (46%)]\tloss=0.2624",
          "Train Epoch: 1 [28160/60000 (47%)]\tloss=0.2127",
          "Train Epoch: 1 [28800/60000 (48%)]\tloss=0.1330",
          "Train Epoch: 1 [29440/60000 (49%)]\tloss=0.2779",
          "Train Epoch: 1 [30080/60000 (50%)]\tloss=0.0940",
          "Train Epoch: 1 [30720/60000 (51%)]\tloss=0.1270",
          "Train Epoch: 1 [31360/60000 (52%)]\tloss=0.2479",
          "Train Epoch: 1 [32000/60000 (53%)]\tloss=0.3388",
          "Train Epoch: 1 [32640/60000 (54%)]\tloss=0.1527",
          "Train Epoch: 1 [33280/60000 (55%)]\tloss=0.0908",
          "Train Epoch: 1 [33920/60000 (57%)]\tloss=0.1445",
          "Train Epoch: 1 [34560/60000 (58%)]\tloss=0.1979",
          "Train Epoch: 1 [35200/60000 (59%)]\tloss=0.2186",
          "Train Epoch: 1 [35840/60000 (60%)]\tloss=0.0632",
          "Train Epoch: 1 [36480/60000 (61%)]\tloss=0.1354",
          "Train Epoch: 1 [37120/60000 (62%)]\tloss=0.1163",
          "Train Epoch: 1 [37760/60000 (63%)]\tloss=0.2355",
          "Train Epoch: 1 [38400/60000 (64%)]\tloss=0.0634",
          "Train Epoch: 1 [39040/60000 (65%)]\tloss=0.1070",
          "Train Epoch: 1 [39680/60000 (66%)]\tloss=0.1599",
          "Train Epoch: 1 [40320/60000 (67%)]\tloss=0.1090",
          "Train Epoch: 1 [40960/60000 (68%)]\tloss=0.1770",
          "Train Epoch: 1 [41600/60000 (69%)]\tloss=0.2301",
          "Train Epoch: 1 [42240/60000 (70%)]\tloss=0.0745",
          "Train Epoch: 1 [42880/60000 (71%)]\tloss=0.1553",
          "Train Epoch: 1 [43520/60000 (72%)]\tloss=0.2793",
          "Train Epoch: 1 [44160/60000 (74%)]\tloss=0.1425",
          "Train Epoch: 1 [44800/60000 (75%)]\tloss=0.1168",
          "Train Epoch: 1 [45440/60000 (76%)]\tloss=0.1225",
          "Train Epoch: 1 [46080/60000 (77%)]\tloss=0.0774",
          "Train Epoch: 1 [46720/60000 (78%)]\tloss=0.1942",
          "Train Epoch: 1 [47360/60000 (79%)]\tloss=0.0686",
          "Train Epoch: 1 [48000/60000 (80%)]\tloss=0.2082",
          "Train Epoch: 1 [48640/60000 (81%)]\tloss=0.1393",
          "Train Epoch: 1 [49280/60000 (82%)]\tloss=0.0940",
          "Train Epoch: 1 [49920/60000 (83%)]\tloss=0.1073",
          "Train Epoch: 1 [50560/60000 (84%)]\tloss=0.1196",
          "Train Epoch: 1 [51200/60000 (85%)]\tloss=0.1445",
          "Train Epoch: 1 [51840/60000 (86%)]\tloss=0.0665",
          "Train Epoch: 1 [52480/60000 (87%)]\tloss=0.0242",
          "Train Epoch: 1 [53120/60000 (88%)]\tloss=0.2633",
          "Train Epoch: 1 [53760/60000 (90%)]\tloss=0.0916",
          "Train Epoch: 1 [54400/60000 (91%)]\tloss=0.1292",
          "Train Epoch: 1 [55040/60000 (92%)]\tloss=0.1909",
          "Train Epoch: 1 [55680/60000 (93%)]\tloss=0.0346",
          "Train Epoch: 1 [56320/60000 (94%)]\tloss=0.0358",
          "Train Epoch: 1 [56960/60000 (95%)]\tloss=0.0767",
          "Train Epoch: 1 [57600/60000 (96%)]\tloss=0.1175",
          "Train Epoch: 1 [58240/60000 (97%)]\tloss=0.1929",
          "Train Epoch: 1 [58880/60000 (98%)]\tloss=0.2051",
          "Train Epoch: 1 [59520/60000 (99%)]\tloss=0.0631",
          "",
          "accuracy=0.9668"
        ]
      }
    ]
  },
  "attributes": {}
}
```

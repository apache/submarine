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
      "resources": "cpu=1,memory=1024M"
    },
    "Worker": {
      "replicas": 1,
      "resources": "cpu=1,memory=2048M"
    }
  }
}
' http://127.0.0.1:32080/api/v1/experiment
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment-1647192232698-0001",
    "uid":"b0ae271b-a01a-43ad-9877-4b8ecbc45de4",
    "status":"Accepted",
    "acceptedTime":"2022-03-14T16:03:10.000+08:00",
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
    "spec":{
      "meta":{
        "experimentId":"experiment-1647192232698-0001",
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{
          "ENV_1":"ENV1"
        },
        "tags":[]
      },
      "environment":{
        "name":null,
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":"apache/submarine:tf-mnist-with-summaries-1.0"
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{
            "memory":"1024M",
            "cpu":"1"
          }
        },
        "Worker":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{
            "memory":"2048M",
            "cpu":"1"
          }
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

### List experiment
`GET /api/v1/experiment`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/experiment
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":[
    {
      "experimentId":"experiment-1647574374688-0002",
      "uid":"cf465781-6310-46d2-92b4-d20161c77d08",
      "status":"Running",
      "acceptedTime":"2022-03-18T15:51:04.000+08:00",
      "createdTime":"2022-03-18T15:51:05.000+08:00",
      "runningTime":"2022-03-18T15:51:17.000+08:00",
      "finishedTime":null,
      "spec":{
        "meta":{
          "experimentId":"experiment-1647574374688-0002",
          "name":"tf-mnist-json",
          "namespace":"default",
          "framework":"TensorFlow",
          "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
          "envVars":{
            "ENV_1":"ENV1"
          },
          "tags":[]
        },
        "environment":{
          "name":null,
          "dockerImage":null,
          "kernelSpec":null,
          "description":null,
          "image":"apache/submarine:tf-mnist-with-summaries-1.0"
        },
        "spec":{
          "Ps":{
            "replicas":1,
            "resources":"cpu\u003d1,memory\u003d1024M",
            "name":null,
            "image":null,
            "cmd":null,
            "envVars":null,
            "resourceMap":{
              "memory":"1024M",
              "cpu":"1"
            }
          },
          "Worker":{
            "replicas":1,
            "resources":"cpu\u003d1,memory\u003d2048M",
            "name":null,
            "image":null,
            "cmd":null,
            "envVars":null,
            "resourceMap":{
              "memory":"2048M",
              "cpu":"1"
            }
          }
        },
        "code":null
      }
    }
  ],
  "attributes":{}
}
```

### Get experiment
`GET /api/v1/experiment/{experiment id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/experiment/experiment-1647574374688-0002
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment-1647574374688-0002",
    "uid":"cf465781-6310-46d2-92b4-d20161c77d08",
    "status":"Running",
    "acceptedTime":"2022-03-18T15:51:04.000+08:00",
    "createdTime":"2022-03-18T15:51:05.000+08:00",
    "runningTime":"2022-03-18T15:51:17.000+08:00",
    "finishedTime":null,
    "spec":{
      "meta":{
        "experimentId":"experiment-1647574374688-0002",
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{
          "ENV_1":"ENV1"
        },
        "tags":[]
      },
      "environment":{
        "name":null,
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":"apache/submarine:tf-mnist-with-summaries-1.0"
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{
            "memory":"1024M",
            "cpu":"1"
          }
        },
        "Worker":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{
            "memory":"2048M",
            "cpu":"1"
          }
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

### Patch experiment
`PATCH /api/v1/experiment/{experiment id}`

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
      "resources": "cpu=1,memory=1024M"
    },
    "Worker": {
      "replicas": 2,
      "resources": "cpu=1,memory=2048M"
    }
  }
}
' http://127.0.0.1:32080/api/v1/experiment/experiment-1647574374688-0002
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment-1647574374688-0002",
    "uid":"b0ae271b-a01a-43ad-9877-4b8ecbc45de4",
    "status":"Succeeded",
    "acceptedTime":"2022-04-04T16:39:25.000+08:00",
    "createdTime":"2022-04-04T16:39:26.000+08:00",
    "runningTime":"2022-04-04T16:39:35.000+08:00",
    "finishedTime":"2022-04-04T16:42:25.000+08:00",
    "spec":{
      "meta":{
        "experimentId":"experiment-1649061491590-0002",
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{
          "ENV_1":"ENV1"
        },
        "tags":[]
      },
      "environment":{
        "name":null,
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":"apache/submarine:tf-mnist-with-summaries-1.0"
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{
            "memory":"1024M",
            "cpu":"1"
          }
        },
        "Worker":{
          "replicas":2,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{
            "memory":"2048M",
            "cpu":"1"
          }
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

### Delete experiment
`GET /api/v1/experiment/{experiment id}`

**Example Request:**
```sh
curl -X DELETE http://127.0.0.1:32080/api/v1/experiment/experiment-1647574374688-0002
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment-1647574374688-0002",
    "uid":"b0ae271b-a01a-43ad-9877-4b8ecbc45de4",
    "status":"Deleted",
    "acceptedTime":null,
    "createdTime":null,
    "runningTime":null,
    "finishedTime":null,
    "spec":{
      "meta":{
        "experimentId":"experiment-1647574374688-0002",
        "name":"tf-mnist-json",
        "namespace":"default",
        "framework":"TensorFlow",
        "cmd":"python /var/tf_mnist/mnist_with_summaries.py --log_dir\u003d/train/log --learning_rate\u003d0.01 --batch_size\u003d150",
        "envVars":{
          "ENV_1":"ENV1"
        },
        "tags":[]
      },
      "environment":{
        "name":null,
        "dockerImage":null,
        "kernelSpec":null,
        "description":null,
        "image":"apache/submarine:tf-mnist-with-summaries-1.0"
      },
      "spec":{
        "Ps":{
          "replicas":1,
          "resources":"cpu\u003d1,memory\u003d1024M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{
            "memory":"1024M",
            "cpu":"1"
          }
        },
        "Worker":{
          "replicas":2,
          "resources":"cpu\u003d1,memory\u003d2048M",
          "name":null,
          "image":null,
          "cmd":null,
          "envVars":null,
          "resourceMap":{
            "memory":"2048M",
            "cpu":"1"
          }
        }
      },
      "code":null
    }
  },
  "attributes":{}
}
```

### List experiment Log
`GET /api/v1/experiment/logs`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/experiment/logs
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":[
    {
      "experimentId":"experiment-1647574374688-0002",
      "logContent":[
        {
          "podName":"experiment-1647574374688-0002-ps-0",
          "podLog":[]
        },
        {
          "podName":"experiment-1647574374688-0002-worker-0",
          "podLog":[
            
          ]
        }
      ]
    }
  ],
  "attributes":{}
}
```

### Get experiment Log
`GET /api/v1/experiment/logs/{id}`

**Example Request:**
```sh
curl -X GET http://127.0.0.1:32080/api/v1/experiment/logs/experiment-1647574374688-0002
```

**Example Response:**
```json
{
  "status":"OK",
  "code":200,
  "success":true,
  "message":null,
  "result":{
    "experimentId":"experiment-1647574374688-0002",
    "logContent":[
      {
        "podName":"experiment-1647574374688-0002-ps-0",
        "podLog":[
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
          "2022-03-18 07:52:07.369276: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA",
          "Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz",
          "Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz",
          "Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz",
          "Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz",
          "Accuracy at step 0: 0.0893",
          "Accuracy at step 10: 0.6851",
          "Accuracy at step 20: 0.8255",
          "Accuracy at step 30: 0.8969",
          "Accuracy at step 40: 0.9009",
          "Accuracy at step 50: 0.9185",
          "Accuracy at step 60: 0.923",
          "Accuracy at step 70: 0.9181",
          "Accuracy at step 80: 0.9344",
          "Accuracy at step 90: 0.9265",
          "Adding run metadata for 99",
          "Accuracy at step 100: 0.9375",
          "Accuracy at step 110: 0.9414",
          "Accuracy at step 120: 0.9402",
          "Accuracy at step 130: 0.9466",
          "Accuracy at step 140: 0.9412",
          "Accuracy at step 150: 0.9497",
          "Accuracy at step 160: 0.9477",
          "Accuracy at step 170: 0.9465",
          "Accuracy at step 180: 0.9546",
          "Accuracy at step 190: 0.9485",
          "Adding run metadata for 199",
          "Accuracy at step 200: 0.9534",
          "Accuracy at step 210: 0.9581",
          "Accuracy at step 220: 0.9418",
          "Accuracy at step 230: 0.9551",
          "Accuracy at step 240: 0.9472",
          "Accuracy at step 250: 0.9555",
          "Accuracy at step 260: 0.9569",
          "Accuracy at step 270: 0.9596",
          "Accuracy at step 280: 0.9588",
          "Accuracy at step 290: 0.9618",
          "Adding run metadata for 299",
          "Accuracy at step 300: 0.9589",
          "Accuracy at step 310: 0.9603",
          "Accuracy at step 320: 0.9632",
          "Accuracy at step 330: 0.956",
          "Accuracy at step 340: 0.9531",
          "Accuracy at step 350: 0.9535",
          "Accuracy at step 360: 0.9517",
          "Accuracy at step 370: 0.9607",
          "Accuracy at step 380: 0.9629",
          "Accuracy at step 390: 0.9553",
          "Adding run metadata for 399",
          "Accuracy at step 400: 0.9623",
          "Accuracy at step 410: 0.9627",
          "Accuracy at step 420: 0.9614",
          "Accuracy at step 430: 0.9604",
          "Accuracy at step 440: 0.9663",
          "Accuracy at step 450: 0.9665",
          "Accuracy at step 460: 0.958",
          "Accuracy at step 470: 0.9643",
          "Accuracy at step 480: 0.9636",
          "Accuracy at step 490: 0.9648",
          "Adding run metadata for 499",
          "Accuracy at step 500: 0.9638",
          "Accuracy at step 510: 0.9629",
          "Accuracy at step 520: 0.9661",
          "Accuracy at step 530: 0.9633",
          "Accuracy at step 540: 0.9669",
          "Accuracy at step 550: 0.9659",
          "Accuracy at step 560: 0.9652",
          "Accuracy at step 570: 0.9675",
          "Accuracy at step 580: 0.9602",
          "Accuracy at step 590: 0.9641",
          "Adding run metadata for 599",
          "Accuracy at step 600: 0.9688",
          "Accuracy at step 610: 0.9638",
          "Accuracy at step 620: 0.9622",
          "Accuracy at step 630: 0.9601",
          "Accuracy at step 640: 0.9636",
          "Accuracy at step 650: 0.9674",
          "Accuracy at step 660: 0.9613",
          "Accuracy at step 670: 0.9706",
          "Accuracy at step 680: 0.9691",
          "Accuracy at step 690: 0.9687",
          "Adding run metadata for 699",
          "Accuracy at step 700: 0.9671",
          "Accuracy at step 710: 0.9659",
          "Accuracy at step 720: 0.9693",
          "Accuracy at step 730: 0.9698",
          "Accuracy at step 740: 0.9681",
          "Accuracy at step 750: 0.9678",
          "Accuracy at step 760: 0.9595",
          "Accuracy at step 770: 0.9697",
          "Accuracy at step 780: 0.9671",
          "Accuracy at step 790: 0.9658",
          "Adding run metadata for 799",
          "Accuracy at step 800: 0.9658",
          "Accuracy at step 810: 0.9702",
          "Accuracy at step 820: 0.9662",
          "Accuracy at step 830: 0.9671",
          "Accuracy at step 840: 0.9731",
          "Accuracy at step 850: 0.9699",
          "Accuracy at step 860: 0.9702",
          "Accuracy at step 870: 0.9686",
          "Accuracy at step 880: 0.9729",
          "Accuracy at step 890: 0.968",
          "Adding run metadata for 899",
          "Accuracy at step 900: 0.9655",
          "Accuracy at step 910: 0.9731",
          "Accuracy at step 920: 0.9676",
          "Accuracy at step 930: 0.9667",
          "Accuracy at step 940: 0.9659",
          "Accuracy at step 950: 0.9689",
          "Accuracy at step 960: 0.9653",
          "Accuracy at step 970: 0.9675",
          "Accuracy at step 980: 0.974",
          "Accuracy at step 990: 0.9723",
          "Adding run metadata for 999"
        ]
      },
      {
        "podName":"experiment-1647574374688-0002-worker-0",
        "podLog":[
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
          "2022-03-18 07:52:07.369085: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA",
          "Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz",
          "Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz",
          "Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz",
          "Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.",
          "Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz",
          "Accuracy at step 0: 0.1348",
          "Accuracy at step 10: 0.7419",
          "Accuracy at step 20: 0.8574",
          "Accuracy at step 30: 0.8959",
          "Accuracy at step 40: 0.9135",
          "Accuracy at step 50: 0.9187",
          "Accuracy at step 60: 0.9276",
          "Accuracy at step 70: 0.9332",
          "Accuracy at step 80: 0.9399",
          "Accuracy at step 90: 0.9376",
          "Adding run metadata for 99",
          "Accuracy at step 100: 0.9378",
          "Accuracy at step 110: 0.9463",
          "Accuracy at step 120: 0.9479",
          "Accuracy at step 130: 0.9468",
          "Accuracy at step 140: 0.9467",
          "Accuracy at step 150: 0.9475",
          "Accuracy at step 160: 0.947",
          "Accuracy at step 170: 0.948",
          "Accuracy at step 180: 0.9472",
          "Accuracy at step 190: 0.954",
          "Adding run metadata for 199",
          "Accuracy at step 200: 0.9492",
          "Accuracy at step 210: 0.9571",
          "Accuracy at step 220: 0.954",
          "Accuracy at step 230: 0.9557",
          "Accuracy at step 240: 0.9557",
          "Accuracy at step 250: 0.9591",
          "Accuracy at step 260: 0.955",
          "Accuracy at step 270: 0.9595",
          "Accuracy at step 280: 0.9596",
          "Accuracy at step 290: 0.9604",
          "Adding run metadata for 299",
          "Accuracy at step 300: 0.9622",
          "Accuracy at step 310: 0.9529",
          "Accuracy at step 320: 0.9609",
          "Accuracy at step 330: 0.9613",
          "Accuracy at step 340: 0.9571",
          "Accuracy at step 350: 0.9599",
          "Accuracy at step 360: 0.9553",
          "Accuracy at step 370: 0.9546",
          "Accuracy at step 380: 0.962",
          "Accuracy at step 390: 0.96",
          "Adding run metadata for 399",
          "Accuracy at step 400: 0.9593",
          "Accuracy at step 410: 0.9641",
          "Accuracy at step 420: 0.9628",
          "Accuracy at step 430: 0.9622",
          "Accuracy at step 440: 0.9639",
          "Accuracy at step 450: 0.9592",
          "Accuracy at step 460: 0.9651",
          "Accuracy at step 470: 0.9658",
          "Accuracy at step 480: 0.9668",
          "Accuracy at step 490: 0.9641",
          "Adding run metadata for 499",
          "Accuracy at step 500: 0.9641",
          "Accuracy at step 510: 0.9561",
          "Accuracy at step 520: 0.9628",
          "Accuracy at step 530: 0.964",
          "Accuracy at step 540: 0.9663",
          "Accuracy at step 550: 0.9681",
          "Accuracy at step 560: 0.968",
          "Accuracy at step 570: 0.967",
          "Accuracy at step 580: 0.9663",
          "Accuracy at step 590: 0.9679",
          "Adding run metadata for 599",
          "Accuracy at step 600: 0.9666",
          "Accuracy at step 610: 0.9648",
          "Accuracy at step 620: 0.9682",
          "Accuracy at step 630: 0.9691",
          "Accuracy at step 640: 0.9683",
          "Accuracy at step 650: 0.966",
          "Accuracy at step 660: 0.9668",
          "Accuracy at step 670: 0.9658",
          "Accuracy at step 680: 0.9709",
          "Accuracy at step 690: 0.9632",
          "Adding run metadata for 699",
          "Accuracy at step 700: 0.9697",
          "Accuracy at step 710: 0.9632",
          "Accuracy at step 720: 0.9641",
          "Accuracy at step 730: 0.9659",
          "Accuracy at step 740: 0.9654",
          "Accuracy at step 750: 0.9694",
          "Accuracy at step 760: 0.968",
          "Accuracy at step 770: 0.9661",
          "Accuracy at step 780: 0.969",
          "Accuracy at step 790: 0.9663",
          "Adding run metadata for 799",
          "Accuracy at step 800: 0.9687",
          "Accuracy at step 810: 0.9651",
          "Accuracy at step 820: 0.9705",
          "Accuracy at step 830: 0.9645",
          "Accuracy at step 840: 0.9652",
          "Accuracy at step 850: 0.9719",
          "Accuracy at step 860: 0.9654",
          "Accuracy at step 870: 0.964",
          "Accuracy at step 880: 0.9645",
          "Accuracy at step 890: 0.9615",
          "Adding run metadata for 899",
          "Accuracy at step 900: 0.9661",
          "Accuracy at step 910: 0.9649",
          "Accuracy at step 920: 0.9569",
          "Accuracy at step 930: 0.9654",
          "Accuracy at step 940: 0.9674",
          "Accuracy at step 950: 0.971",
          "Accuracy at step 960: 0.9684",
          "Accuracy at step 970: 0.9648",
          "Accuracy at step 980: 0.9693",
          "Accuracy at step 990: 0.9627",
          "Adding run metadata for 999"
        ]
      }
    ]
  },
  "attributes":{}
}
```

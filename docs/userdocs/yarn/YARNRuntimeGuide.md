<!--
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-->

# YARN Runtime Quick Start Guide

## Prerequisite

Check out the [Readme](README.md)

## Build your own Docker image

When you follow the documents below, and want to build your own Docker image for Tensorflow/PyTorch/MXNet? Please check out [Build your Docker image](Dockerfiles.md) for more details.

## Launch TensorFlow Application:

### Without Docker

You need:

* Build a Python virtual environment with TensorFlow 1.13.1 installed
* A cluster with Hadoop 2.9 or above.

### Building a Python virtual environment with TensorFlow

TonY requires a Python virtual environment zip with TensorFlow and any needed Python libraries already installed.

```
wget https://files.pythonhosted.org/packages/33/bc/fa0b5347139cd9564f0d44ebd2b147ac97c36b2403943dbee8a25fd74012/virtualenv-16.0.0.tar.gz
tar xf virtualenv-16.0.0.tar.gz

# Make sure to install using Python 3, as TensorFlow only provides Python 3 artifacts
python virtualenv-16.0.0/virtualenv.py venv
. venv/bin/activate
pip install tensorflow==1.13.1
zip -r myvenv.zip venv
deactivate
```

The above commands will produced a myvenv.zip and it will be used in below example. There's no need to copy it to other nodes. And it is not needed when using Docker to run the job.


**Note:** If you require a version of TensorFlow and TensorBoard prior to `1.13.1`, take a look at [this](https://github.com/linkedin/TonY/issues/42) issue.


### Get the training examples

Get mnist_distributed.py from https://github.com/linkedin/TonY/tree/master/tony-examples/mnist-tensorflow


```
SUBMARINE_VERSION=0.4.0-SNAPSHOT
SUBMARINE_HADOOP_VERSION=3.1
CLASSPATH=$(hadoop classpath --glob):path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar \
java org.apache.submarine.client.cli.Cli job run --name tf-job-001 \
 --framework tensorflow \
 --verbose \
 --input_path "" \
 --num_workers 2 \
 --worker_resources memory=1G,vcores=1 \
 --num_ps 1 \
 --ps_resources memory=1G,vcores=1 \
 --worker_launch_cmd "myvenv.zip/venv/bin/python mnist_distributed.py --steps 2 --data_dir /tmp/data --working_dir /tmp/mode" \
 --ps_launch_cmd "myvenv.zip/venv/bin/python mnist_distributed.py --steps 2 --data_dir /tmp/data --working_dir /tmp/mode" \
 --insecure \
 --conf tony.containers.resources=path-to/myvenv.zip#archive,path-to/mnist_distributed.py,path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar
```
You should then be able to see links and status of the jobs from command line:

```
2019-04-22 20:30:42,611 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: worker index: 0 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000003/pi status: RUNNING
2019-04-22 20:30:42,612 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: worker index: 1 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000004/pi status: RUNNING
2019-04-22 20:30:42,612 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: ps index: 0 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000002/pi status: RUNNING
2019-04-22 20:30:42,612 INFO tony.TonyClient: Logs for ps 0 at: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000002/pi
2019-04-22 20:30:42,612 INFO tony.TonyClient: Logs for worker 0 at: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000003/pi
2019-04-22 20:30:42,612 INFO tony.TonyClient: Logs for worker 1 at: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000004/pi
2019-04-22 20:30:44,625 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: ps index: 0 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000002/pi status: FINISHED
2019-04-22 20:30:44,625 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: worker index: 0 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000003/pi status: FINISHED
2019-04-22 20:30:44,626 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: worker index: 1 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000004/pi status: FINISHED

```

### With Docker

```
SUBMARINE_VERSION=0.4.0-SNAPSHOT
SUBMARINE_HADOOP_VERSION=3.1
CLASSPATH=$(hadoop classpath --glob):path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar \
java org.apache.submarine.client.cli.Cli job run --name tf-job-001 \
 --framework tensorflow \
 --docker_image hadoopsubmarine/tf-1.8.0-cpu:0.0.1 \
 --input_path hdfs://pi-aw:9000/dataset/cifar-10-data \
 --worker_resources memory=3G,vcores=2 \
 --worker_launch_cmd "export CLASSPATH=\$(/hadoop-3.1.0/bin/hadoop classpath --glob) && cd /test/models/tutorials/image/cifar10_estimator && python cifar10_main.py --data-dir=%input_path% --job-dir=%checkpoint_path% --train-steps=10000 --eval-batch-size=16 --train-batch-size=16 --variable-strategy=CPU --num-gpus=0 --sync" \
 --env JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
 --env DOCKER_HADOOP_HDFS_HOME=/hadoop-3.1.0 \
 --env DOCKER_JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
 --env HADOOP_HOME=/hadoop-3.1.0 \
 --env HADOOP_YARN_HOME=/hadoop-3.1.0 \
 --env HADOOP_COMMON_HOME=/hadoop-3.1.0 \
 --env HADOOP_HDFS_HOME=/hadoop-3.1.0 \
 --env HADOOP_CONF_DIR=/hadoop-3.1.0/etc/hadoop \
 --conf tony.containers.resources=path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar
```
#### Notes:
1) `DOCKER_JAVA_HOME` points to JAVA_HOME inside Docker image.

2) `DOCKER_HADOOP_HDFS_HOME` points to HADOOP_HDFS_HOME inside Docker image.

We removed TonY submodule after applying [SUBMARINE-371](https://issues.apache.org/jira/browse/SUBMARINE-371) and changed to use TonY dependency directly.

After Submarine v0.2.0, there is a uber jar `submarine-all-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}.jar` released together with
the `submarine-core-${SUBMARINE_VERSION}.jar`, `submarine-yarnservice-runtime-${SUBMARINE_VERSION}.jar` and `submarine-tony-runtime-${SUBMARINE_VERSION}.jar`.
<br />

## Launch PyTorch Application:

### Without Docker

You need:

* Build a Python virtual environment with PyTorch 0.4.0+ installed
* A cluster with Hadoop 2.9 or above.

### Building a Python virtual environment with PyTorch

TonY requires a Python virtual environment zip with PyTorch and any needed Python libraries already installed.

```
wget https://files.pythonhosted.org/packages/33/bc/fa0b5347139cd9564f0d44ebd2b147ac97c36b2403943dbee8a25fd74012/virtualenv-16.0.0.tar.gz
tar xf virtualenv-16.0.0.tar.gz

python virtualenv-16.0.0/virtualenv.py venv
. venv/bin/activate
pip install pytorch==0.4.0
zip -r myvenv.zip venv
deactivate
```


### Get the training examples

Get mnist_distributed.py from https://github.com/linkedin/TonY/tree/master/tony-examples/mnist-pytorch


```
SUBMARINE_VERSION=0.4.0-SNAPSHOT
SUBMARINE_HADOOP_VERSION=3.1
CLASSPATH=$(hadoop classpath --glob):path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar \
java org.apache.submarine.client.cli.Cli job run --name PyTorch-job-001 \
 --framework pytorch
 --num_workers 2 \
 --worker_resources memory=3G,vcores=2 \
 --num_ps 2 \
 --ps_resources memory=3G,vcores=2 \
 --worker_launch_cmd "myvenv.zip/venv/bin/python mnist_distributed.py" \
 --ps_launch_cmd "myvenv.zip/venv/bin/python mnist_distributed.py" \
 --insecure \
 --conf tony.containers.resources=path-to/myvenv.zip#archive,path-to/mnist_distributed.py, \
path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar
```
You should then be able to see links and status of the jobs from command line:

```
2019-04-22 20:30:42,611 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: worker index: 0 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000003/pi status: RUNNING
2019-04-22 20:30:42,612 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: worker index: 1 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000004/pi status: RUNNING
2019-04-22 20:30:42,612 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: ps index: 0 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000002/pi status: RUNNING
2019-04-22 20:30:42,612 INFO tony.TonyClient: Logs for ps 0 at: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000002/pi
2019-04-22 20:30:42,612 INFO tony.TonyClient: Logs for worker 0 at: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000003/pi
2019-04-22 20:30:42,612 INFO tony.TonyClient: Logs for worker 1 at: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000004/pi
2019-04-22 20:30:44,625 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: ps index: 0 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000002/pi status: FINISHED
2019-04-22 20:30:44,625 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: worker index: 0 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000003/pi status: FINISHED
2019-04-22 20:30:44,626 INFO tony.TonyClient: Tasks Status Updated: [TaskInfo] name: worker index: 1 url: http://pi-aw:8042/node/containerlogs/container_1555916523933_0030_01_000004/pi status: FINISHED

```

### With Docker

```
SUBMARINE_VERSION=0.4.0-SNAPSHOT
SUBMARINE_HADOOP_VERSION=3.1
CLASSPATH=$(hadoop classpath --glob):path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar \
java org.apache.submarine.client.cli.Cli job run --name PyTorch-job-001 \
 --framework pytorch
 --docker_image pytorch-latest-gpu:0.0.1 \
 --input_path "" \
 --num_workers 1 \
 --worker_resources memory=3G,vcores=2 \
 --worker_launch_cmd "cd /test/ && python cifar10_tutorial.py" \
 --env JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
 --env DOCKER_HADOOP_HDFS_HOME=/hadoop-3.1.2 \
 --env DOCKER_JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
 --env HADOOP_HOME=/hadoop-3.1.2 \
 --env HADOOP_YARN_HOME=/hadoop-3.1.2 \
 --env HADOOP_COMMON_HOME=/hadoop-3.1.2 \
 --env HADOOP_HDFS_HOME=/hadoop-3.1.2 \
 --env HADOOP_CONF_DIR=/hadoop-3.1.2/etc/hadoop \
 --conf tony.containers.resources=path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar
```

## Launch MXNet Application:

### Without Docker

You need:

* Build a Python virtual environment with MXNet installed
* A cluster with Hadoop 2.9 or above.

### Building a Python virtual environment with MXNet

TonY requires a Python virtual environment zip with MXNet and any needed Python libraries already installed.

```
wget https://files.pythonhosted.org/packages/33/bc/fa0b5347139cd9564f0d44ebd2b147ac97c36b2403943dbee8a25fd74012/virtualenv-16.0.0.tar.gz
tar xf virtualenv-16.0.0.tar.gz

python virtualenv-16.0.0/virtualenv.py venv
. venv/bin/activate
pip install mxnet==1.5.1
zip -r myvenv.zip venv
deactivate
```


### Get the training examples

Get image_classification.py from this [link](https://github.com/apache/submarine/blob/master/dev-support/mini-submarine/submarine/image_classification.py)


```
SUBMARINE_VERSION=0.4.0-SNAPSHOT
SUBMARINE_HADOOP_VERSION=3.1
CLASSPATH=$(hadoop classpath --glob):path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar \
java org.apache.submarine.client.cli.Cli job run --name MXNet-job-001 \
 --framework mxnet
 --input_path "" \
 --num_workers 2 \
 --worker_resources memory=3G,vcores=2 \
 --worker_launch_cmd "myvenv.zip/venv/bin/python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync" \
 --num_ps 2 \
 --ps_resources memory=3G,vcores=2 \
 --ps_launch_cmd "myvenv.zip/venv/bin/python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync" \
 --num_schedulers=1 \
 --scheduler_resources memory=1G,vcores=1 \
 --scheduler_launch_cmd="myvenv.zip/venv/bin/python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync" \
 --insecure \
 --conf tony.containers.resources=path-to/myvenv.zip#archive,path-to/image_classification.py, \
path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar
```
You should then be able to see links and status of the jobs from command line:

```
2020-04-16 20:23:43,834 INFO tony.TonyClient: Task status updated: [TaskInfo] name: server, index: 1, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000004/pi status: RUNNING
2020-04-16 20:23:43,834 INFO tony.TonyClient: Task status updated: [TaskInfo] name: server, index: 0, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000003/pi status: RUNNING
2020-04-16 20:23:43,834 INFO tony.TonyClient: Task status updated: [TaskInfo] name: worker, index: 1, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000006/pi status: RUNNING
2020-04-16 20:23:43,834 INFO tony.TonyClient: Task status updated: [TaskInfo] name: worker, index: 0, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000005/pi status: RUNNING
2020-04-16 20:23:43,834 INFO tony.TonyClient: Task status updated: [TaskInfo] name: scheduler, index: 0, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000002/pi status: RUNNING
2020-04-16 20:23:43,839 INFO tony.TonyClient: Logs for scheduler 0 at: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000002/pi
2020-04-16 20:23:43,839 INFO tony.TonyClient: Logs for server 0 at: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000003/pi
2020-04-16 20:23:43,840 INFO tony.TonyClient: Logs for server 1 at: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000004/pi
2020-04-16 20:23:43,840 INFO tony.TonyClient: Logs for worker 0 at: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000005/pi
2020-04-16 20:23:43,840 INFO tony.TonyClient: Logs for worker 1 at: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000006/pi
2020-04-16 21:02:09,723 INFO tony.TonyClient: Task status updated: [TaskInfo] name: scheduler, index: 0, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000002/pi status: SUCCEEDED
2020-04-16 21:02:09,736 INFO tony.TonyClient: Task status updated: [TaskInfo] name: worker, index: 0, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000005/pi status: SUCCEEDED
2020-04-16 21:02:09,737 INFO tony.TonyClient: Task status updated: [TaskInfo] name: server, index: 1, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000004/pi status: SUCCEEDED
2020-04-16 21:02:09,737 INFO tony.TonyClient: Task status updated: [TaskInfo] name: worker, index: 1, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000006/pi status: SUCCEEDED
2020-04-16 21:02:09,737 INFO tony.TonyClient: Task status updated: [TaskInfo] name: server, index: 0, url: http://pi-aw:8042/node/containerlogs/container_1587037749540_0005_01_000003/pi status: SUCCEEDED
```

### With Docker
You could refer to this [sample Dockerfile](project/github/submarine/docs/userdocs/yarn/docker/mxnet/cifar10/Dockerfile.cifar10.mx_1.5.1) for building your own Docker image.
```
SUBMARINE_VERSION=0.4.0-SNAPSHOT
SUBMARINE_HADOOP_VERSION=3.1
CLASSPATH=$(hadoop classpath --glob):path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar \
java org.apache.submarine.client.cli.Cli job run --name MXNet-job-001 \
 --framework mxnet
 --docker_image <your_docker_image> \
 --input_path "" \
 --num_schedulers 1 \
 --scheduler_resources memory=1G,vcores=1 \
 --scheduler_launch_cmd "/usr/bin/python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync" \
 --num_workers 2 \
 --worker_resources memory=2G,vcores=1 \
 --worker_launch_cmd "/usr/bin/python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync" \
 --num_ps 2 \
 --ps_resources memory=2G,vcores=1 \
 --ps_launch_cmd "/usr/bin/python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync" \
 --verbose \
 --insecure \
 --conf tony.containers.resources=path-to/image_classification.py,path-to/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar
```

## Use YARN Service to run Submarine: Deprecated

Historically, Submarine supports to use [YARN Service](https://hadoop.apache.org/docs/r3.1.0/hadoop-yarn/hadoop-yarn-site/yarn-service/Overview.html) to submit deep learning jobs. Now we stop supporting it because YARN service is not actively developed by community, and extra dependencies such as RegistryDNS/ATS-v2 causes lots of issues for setup.

As of now, you can still use YARN service to run Submarine, but code will be removed in the future release. We will only support use TonY when use Submarine on YARN.

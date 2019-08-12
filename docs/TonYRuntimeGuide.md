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

# TonY Runtime Quick Start Guide

## Prerequisite

Check out the [QuickStart](QuickStart.md)

## Launch TensorFlow Application:

### Without Docker

You need:

* Build a Python virtual environment with TensorFlow 1.13.1 installed
* A cluster with Hadoop 2.7 or above.
* TonY library 0.3.2 or above. Download from [here](https://github.com/linkedin/TonY/releases)


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
SUBMARINE_VERSION=0.2.0
SUBMARINE_HOME=path-to/hadoop-submarine-dist-0.2.0-hadoop-3.1
CLASSPATH=$(hadoop classpath --glob):${SUBMARINE_HOME}/hadoop-submarine-core-${SUBMARINE_VERSION}.jar:${SUBMARINE_HOME}/hadoop-submarine-tony-runtime-${SUBMARINE_VERSION}.jar:path-to/tony-cli-0.3.13-all.jar \
java org.apache.hadoop.yarn.submarine.client.cli.Cli job run --name tf-job-001 \
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
 --conf tony.containers.resources=path-to/myvenv.zip#archive,path-to/mnist_distributed.py,path-to/tony-cli-0.3.13-all.jar

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
SUBMARINE_VERSION=0.2.0
SUBMARINE_HOME=path-to/hadoop-submarine-dist-0.2.0-hadoop-3.1
CLASSPATH=$(hadoop classpath --glob):${SUBMARINE_HOME}/hadoop-submarine-core-${SUBMARINE_VERSION}.jar:${SUBMARINE_HOME}/hadoop-submarine-tony-runtime-${SUBMARINE_VERSION}.jar:path-to/tony-cli-0.3.13-all.jar \
java org.apache.hadoop.yarn.submarine.client.cli.Cli job run --name tf-job-001 \
 --framework tensorflow \
 --docker_image hadoopsubmarine/tf-1.8.0-cpu:0.0.3 \
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
 --conf tony.containers.resources=/home/pi/hadoop/TonY/tony-cli/build/libs/tony-cli-0.3.2-all.jar
```
#### Notes:
1) `DOCKER_JAVA_HOME` points to JAVA_HOME inside Docker image.

2) `DOCKER_HADOOP_HDFS_HOME` points to HADOOP_HDFS_HOME inside Docker image.

After Submarine v0.2.0, there is a uber jar `hadoop-submarine-all-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}.jar` released together with
the `hadoop-submarine-core-${SUBMARINE_VERSION}.jar`, `hadoop-submarine-yarnservice-runtime-${SUBMARINE_VERSION}.jar` and `hadoop-submarine-tony-runtime-${SUBMARINE_VERSION}.jar`.
<br />
To use the uber jar, we need to replace the three jars in previous command with one uber jar and put hadoop config directory in CLASSPATH.

## Launch PyToch Application:

### Without Docker

You need:

* Build a Python virtual environment with PyTorch 0.4.0+ installed


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
SUBMARINE_VERSION=0.2.0
SUBMARINE_HOME=path-to/hadoop-submarine-dist-0.2.0-hadoop-3.1
CLASSPATH=$(hadoop classpath --glob):${SUBMARINE_HOME}/hadoop-submarine-core-${SUBMARINE_VERSION}.jar:${SUBMARINE_HOME}/hadoop-submarine-tony-runtime-${SUBMARINE_VERSION}.jar:path-to/tony-cli-0.3.13-all.jar \
java org.apache.hadoop.yarn.submarine.client.cli.Cli job run --name tf-job-001 \
 --num_workers 2 \
 --worker_resources memory=3G,vcores=2 \
 --num_ps 2 \
 --ps_resources memory=3G,vcores=2 \
 --worker_launch_cmd "myvenv.zip/venv/bin/python mnist_distributed.py" \
 --ps_launch_cmd "myvenv.zip/venv/bin/python mnist_distributed.py" \
 --insecure \
 --conf tony.containers.resources=PATH_TO_VENV_YOU_CREATED/myvenv.zip#archive,PATH_TO_MNIST_EXAMPLE/mnist_distributed.py, \
PATH_TO_TONY_CLI_JAR/tony-cli-0.3.2-all.jar \
--conf tony.application.framework=pytorch

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
SUBMARINE_VERSION=0.2.0
SUBMARINE_HOME=path-to/hadoop-submarine-dist-0.2.0-hadoop-3.1
CLASSPATH=$(hadoop classpath --glob):${SUBMARINE_HOME}/hadoop-submarine-core-${SUBMARINE_VERSION}.jar:${SUBMARINE_HOME}/hadoop-submarine-tony-runtime-${SUBMARINE_VERSION}.jar:path-to/tony-cli-0.3.13-all.jar \
java org.apache.hadoop.yarn.submarine.client.cli.Cli job run --name tf-job-001 \
 --docker_image hadoopsubmarine/tf-1.8.0-cpu:0.0.3 \
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
 --conf tony.containers.resources=PATH_TO_TONY_CLI_JAR/tony-cli-0.3.2-all.jar \
 --conf tony.application.framework=pytorch
```


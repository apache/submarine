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

# Creating Docker Images for Running MXNet on YARN

## How to create docker images to run MXNet on YARN

Dockerfile to run MXNet on YARN needs two parts:

**Base libraries which MXNet depends on**

1) OS base image, for example ```ubuntu:18.04```

2) MXNet dependent libraries and packages. \
   For example ```python```, ```scipy```. For GPU support, you also need ```cuda```, ```cudnn```, etc.

3) MXNet package.

**Libraries to access HDFS**

1) JDK

2) Hadoop

Here's an example of a base image (without GPU support) to install MXNet:
```shell
FROM ubuntu:18.04

# Install some development tools and packages
# MXNet 1.6 is going to be the last MXNet release to support Python2
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata git \
    wget zip python3 python3-pip python3-distutils libgomp1 libopenblas-dev libopencv-dev

# Install latest MXNet using pip (without GPU support)
RUN pip3 install mxnet

RUN echo "Install python related packages" && \
    pip3 install --user graphviz==0.8.4 ipykernel jupyter matplotlib numpy pandas scipy sklearn  && \
    python3 -m ipykernel.kernelspec
```

On top of above image, add files, install packages to access HDFS
```shell
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
RUN apt-get update && apt-get install -y openjdk-8-jdk wget

# Install hadoop
ENV HADOOP_VERSION="3.1.2"
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz
# If you are in mainland China, you can use the following command.
# RUN wget http://mirrors.hust.edu.cn/apache/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz

RUN tar zxf hadoop-${HADOOP_VERSION}.tar.gz
RUN ln -s hadoop-${HADOOP_VERSION} hadoop-current
RUN rm hadoop-${HADOOP_VERSION}.tar.gz
```

Build and push to your own docker registry: Use ```docker build ... ``` and ```docker push ...``` to finish this step.

## Use examples to build your own MXNet docker images

We provided some example Dockerfiles for you to build your own MXNet docker images.

For latest MXNet

- *docker/mxnet/base/ubuntu-18.04/Dockerfile.cpu.mxnet_latest*: Latest MXNet that supports CPU
- *docker/mxnet/base/ubuntu-18.04/Dockerfile.gpu.mxnet_latest*: Latest MXNet that supports GPU, which is prebuilt to CUDA10.

# Build Docker images

### Manually build Docker image:

Under `docker/mxnet` directory, run `build-all.sh` to build all Docker images. This command will build the following Docker images:

- `mxnet-latest-cpu-base:0.0.1` for base Docker image which includes Hadoop, MXNet
- `mxnet-latest-gpu-base:0.0.1` for base Docker image which includes Hadoop, MXNet, GPU base libraries.

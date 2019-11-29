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

## Creating Docker Images for Running Kaldi on YARN

### How to create docker images to run Kaldi on YARN

Dockerfile to run Kaldi on YARN need two part:

**Base libraries which Kaldi depends on**

1) OS base image, for example ```nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04```

2) Kaldi depended libraries and packages. For example ```python```, ```g++```, ```make```. For GPU support, need ```cuda```, ```cudnn```, etc.

3) Kaldi compile.

**Libraries to access HDFS**

1) JDK

2) Hadoop

Here's an example of a base image (w/o GPU support) to install Kaldi:
```shell
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        openjdk-8-jdk \
        iputils-ping \
        g++ \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        python2.7 \
        python3 \
        zlib1g-dev \
        ca-certificates \
        patch \
        ffmpeg \
        vim && \
        rm -rf /var/lib/apt/lists/* && \
        ln -s /usr/bin/python2.7 /usr/bin/python

RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
    cd /opt/kaldi && \
    cd /opt/kaldi/tools && \
    ./extras/install_mkl.sh && \
    make -j $(nproc) && \
    cd /opt/kaldi/src && \
    ./configure --shared --use-cuda && \
    make depend -j $(nproc) && \
    make -j $(nproc)
```

On top of above image, add files, install packages to access HDFS
```shell
RUN apt-get update && apt-get install -y openjdk-8-jdk wget
# Install hadoop
ENV HADOOP_VERSION="3.2.1"
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar zxf hadoop-${HADOOP_VERSION}.tar.gz && \
    ln -s hadoop-${HADOOP_VERSION} hadoop-current && \
    rm hadoop-${HADOOP_VERSION}.tar.gz
```

Build and push to your own docker registry: Use ```docker build ... ``` and ```docker push ...``` to finish this step.

### Use examples to build your own Kaldi docker images

We provided following examples for you to build kaldi docker images.

For latest Kaldi

- *base/ubuntu-16.04/Dockerfile.gpu.kaldi_latest: Latest Kaldi that supports GPU, which is prebuilt to CUDA10, with models.

### Build Docker images

#### Manually build Docker image:

Under `docker/` directory,The CLUSTER_NAME can be modified in build-all.sh to have installation permissions, run `build-all.sh` to build Docker images. It will build following images:

- `kaldi-latest-gpu-base:0.0.1` for base Docker image which includes Hadoop, Kaldi, GPU base libraries, which includes thchs30 model.

#### Use prebuilt images

(No liability)
You can also use prebuilt images for convenience in the docker hub:
- hadoopsubmarine/kaldi-latest-gpu-base:0.0.1

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
# Install JAVA
RUN apt-get -q update \
    && apt-get -q install -y --no-install-recommends openjdk-8-jdk libbcprov-java \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Docker
RUN apt-get update && \
  apt-get -y install apt-transport-https ca-certificates curl software-properties-common && \
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && \
  add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && \
  apt-get update && \
  apt-get -y install docker-ce
# So no need to mount host's /var/run/docker.sock, dockerd will create in local fs
VOLUME /var/lib/docker
VOLUME /var/lib/docker.sock

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV JRE_HOME /usr/lib/jvm/java-8-openjdk-amd64/jre

# Install user tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        git make libgtest-dev cmake wget unzip libtinfo-dev libz-dev \
        libcurl4-openssl-dev libopenblas-dev g++ \
        apt-transport-https curl vim ca-certificates maven

WORKDIR /root

# Intstall conda
ADD install-conda.sh /usr/local
ADD bootstrap.sh /usr/local
RUN chmod 755 /usr/local/install-conda.sh && \
    bash /usr/local/install-conda.sh

ENV PATH=${PATH}:/root/miniconda/bin

WORKDIR /workspace

ENTRYPOINT [ "/usr/local/bootstrap.sh" ]

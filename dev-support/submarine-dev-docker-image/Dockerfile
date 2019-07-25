# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:16.04
#INSTALL JAVA
RUN apt-get -q update \
    && apt-get -q install -y --no-install-recommends openjdk-8-jdk libbcprov-java \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/jre

#INSTALL Docker
RUN \
  apt-get update && \
  apt-get -y install apt-transport-https ca-certificates curl software-properties-common && \
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && \
  add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" && \
  apt-get update && \
  apt-get -y install docker-ce
# So no need to mount host's /var/run/docker.sock, dockerd will create in local fs
VOLUME /var/lib/docker

#INSTALL user tools
RUN \
  apt-get update && \
  apt-get -y install vim

#INSTALL HADOOP
# Add native libs
ARG HADOOP_VERSION=
ADD hadoop-${HADOOP_VERSION}.tar.gz /usr/local
#ADD hadoop-native-${HADOOP_VERSION}.tar /usr/local/hadoop-${HADOOP_VERSION}/lib/native

ENV HADOOP_PREFIX=/usr/local/hadoop \
    HADOOP_COMMON_HOME=/usr/local/hadoop \
    HADOOP_HDFS_HOME=/usr/local/hadoop \
    HADOOP_MAPRED_HOME=/usr/local/hadoop \
    HADOOP_YARN_HOME=/usr/local/hadoop \
    HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop \
    YARN_CONF_DIR=/usr/local/hadoop/etc/hadoop \
    PATH=${PATH}:/usr/local/hadoop/bin

RUN \
  cd /usr/local && mv ./hadoop-${HADOOP_VERSION} hadoop && \
  rm -f ${HADOOP_PREFIX}/logs/*

ARG ZK_VERSION=3.4.14
ADD zookeeper-${ZK_VERSION}.tar.gz /usr/local
RUN mv /usr/local/zookeeper-${ZK_VERSION} /usr/local/zookeeper
RUN sed "s#/tmp/zookeeper#/tmp/staging/zookeeper#" /usr/local/zookeeper/conf/zoo_sample.cfg > /usr/local/zookeeper/conf/zoo.cfg

WORKDIR $HADOOP_PREFIX

# Hdfs ports
EXPOSE 50010 50020 50070 50075 50090 8020 9000
# Mapred ports
EXPOSE 19888
#Yarn ports
EXPOSE 8030 8031 8032 8033 8040 8042 8088
# ZK ports
EXPOSE 2181 2888 3888
#Other ports
EXPOSE 49707 2122

#Add spark dynamic allocation jar
#ADD spark-2.4.0-yarn-shuffle.jar /usr/local/hadoop/share/hadoop/yarn/spark-2.4.0-yarn-shuffle.jar

# Create users
RUN \
  groupadd -g 1007 hadoop && \
  useradd -m -G hadoop -u 1008 -s /bin/bash yarn && \
  chown -R root:hadoop /usr/local/hadoop && \
  chown -R yarn:hadoop /usr/local/zookeeper

# Copy Config
COPY conf /tmp/hadoop-config

ENV HADOOP_VER=${HADOOP_VERSION}
#Install Spark
ARG SPARK_VERSION=
ENV SPARK_VER=${SPARK_VERSION}
ADD spark-${SPARK_VERSION}-bin-hadoop2.7.tgz /opt
ADD spark-defaults-dynamic-allocation.conf /opt/spark-${SPARK_VERSION}/conf/spark-defaults.conf
RUN \
  apt-get update && \
  apt-get install -y vim python python-numpy wget zip python3
# Add pyspark sample
ADD spark-script /home/yarn/spark-script
RUN chown -R yarn /home/yarn/spark-script && \
    chmod +x -R /home/yarn/spark-script
# Add distributedShell example
ADD conf/yarn-ds-docker.sh /home/yarn
RUN chown -R yarn /home/yarn/yarn-ds-docker.sh && \
    chmod +x /home/yarn/yarn-ds-docker.sh

# set image name env
ARG IMAGE_NAME=
ENV IMAGE_N=${IMAGE_NAME}

# Install Submarine
ARG SUBMARINE_VERSION=
ENV SUBMARINE_VER=${SUBMARINE_VERSION}
ADD hadoop-submarine-${SUBMARINE_VER}.tar.gz /opt
ADD submarine /home/yarn/submarine

# Build virtual python env
RUN cd /home/yarn/submarine && \
    chmod +x /home/yarn/submarine/* && \
    /home/yarn/submarine/build_python_virtual_env.sh

# Grant read permission for submarine job
RUN chown yarn /home/yarn/submarine

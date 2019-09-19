#!/bin/bash
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

hadoop_v=2.9.2
spark_v=2.4.4
submarine_v=0.2.0
image_name="local/mini-submarine:${submarine_v}"

download_package() {
  if [ -f "$1" ]; then
    echo "Found $1"
  else
    echo "Start downloading the package $1 from $2"
    if type wget >/dev/null 2>&1; then
      wget $2/$1
    elif type curl >/dev/null 2>&1; then
      curl -O $2/$1
    else
      echo 'We need a tool to transfer data from or to a server. Such as wget/curl.'
      echo 'Bye, bye!'
      exit -1
    fi
  fi
}

# download hadoop
download_package "hadoop-${hadoop_v}.tar.gz" "http://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-${hadoop_v}"
# download spark
download_package "spark-${spark_v}-bin-hadoop2.7.tgz" "http://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-${spark_v}"
# download zookeeper
download_package "zookeeper-3.4.14.tar.gz" "http://mirror.bit.edu.cn/apache/zookeeper/zookeeper-3.4.14"
# download submarine
download_package "hadoop-submarine-${submarine_v}.tar.gz" "http://mirror.bit.edu.cn/apache/hadoop/submarine/submarine-${submarine_v}"

# build image
echo "Start building the mini-submarine docker image..."
docker build --build-arg HADOOP_VERSION=${hadoop_v} --build-arg SPARK_VERSION=${spark_v} --build-arg SUBMARINE_VERSION=${submarine_v} --build-arg IMAGE_NAME=${image_name} -t ${image_name} .

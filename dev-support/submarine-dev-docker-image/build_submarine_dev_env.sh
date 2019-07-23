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
spark_v=2.4.3
submarine_v=0.2.0
image_name="local/hadoop-docker:submarine"

# download hadoop
file="hadoop-${hadoop_v}.tar.gz"
if [ -e "$file" ]
then
  echo "$file found."
else
  echo "$file not found."
  wget http://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-${hadoop_v}/hadoop-${hadoop_v}.tar.gz
fi
# download spark
file2="spark-${spark_v}-bin-hadoop2.7.tgz"
if [ -e "$file2" ]
then
  echo "$file2 found."
else
  echo "$file2 not found."
  wget http://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-${spark_v}/spark-${spark_v}-bin-hadoop2.7.tgz
fi

# download zookeeper
file3="zookeeper-3.4.14.tar.gz"
if [ -e "$file3" ]
then
  echo "$file3 found."
else
  echo "$file3 not found."
  wget http://mirror.bit.edu.cn/apache/zookeeper/zookeeper-3.4.14/zookeeper-3.4.14.tar.gz
fi


# download submarine
file4="hadoop-submarine-${submarine_v}.tar.gz"
if [ -e "$file4" ]
then
  echo "$file4 found."
else
  echo "$file4 not found."
  wget http://mirror.bit.edu.cn/apache/hadoop/submarine/submarine-${submarine_v}/hadoop-submarine-${submarine_v}.tar.gz
fi


# build image
docker build --build-arg HADOOP_VERSION=${hadoop_v} --build-arg SPARK_VERSION=${spark_v} --build-arg SUBMARINE_VERSION=${submarine_v} --build-arg IMAGE_NAME=${image_name} -t ${image_name} .


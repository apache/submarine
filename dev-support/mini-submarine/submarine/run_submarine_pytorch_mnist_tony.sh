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

while [ $# -gt 0 ]; do
  case "$1" in
    --debug*)
      DEBUG=$1
      if [ -n "$2" ]; then
        DEBUG_PORT=$2
        shift
      fi
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ "$DEBUG" ]; then
  if [ -z "$DEBUG_PORT" ]; then
    DEBUG_PORT=8000
  fi
  JAVA_CMD="java -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=${DEBUG_PORT}"
else
  JAVA_CMD="java"
fi

SUBMARINE_VERSION=0.3.0-SNAPSHOT
HADOOP_VERSION=2.9
SUBMARINE_PATH=/opt/submarine-current
HADOOP_CONF_PATH=/usr/local/hadoop/etc/hadoop
MNIST_PATH=/home/yarn/submarine

${JAVA_CMD} -cp ${SUBMARINE_PATH}/submarine-all-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}.jar:${HADOOP_CONF_PATH} \
 org.apache.submarine.client.cli.Cli job run \
 --name pytorch-job-001 \
 --framework pytorch \
 --input_path "" \
 --num_workers 2 \
 --worker_resources memory=1G,vcores=1 \
 --worker_launch_cmd "myvenv.zip/venv/bin/python pytorch_mnist_distributed.py" \
 --insecure \
 --verbose \
 --conf tony.containers.resources=${MNIST_PATH}/myvenv.zip#archive,${MNIST_PATH}/pytorch_mnist_distributed.py,${SUBMARINE_PATH}/submarine-all-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}.jar

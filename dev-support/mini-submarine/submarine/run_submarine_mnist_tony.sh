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

#!/bin/bash

# Below are configurable variables, please adapt base on your local environment.
# Version of submarine jar
SUBMARINE_VERSION=0.3.0-SNAPSHOT

# Version of affiliated Hadoop version for this Submarine jar.
SUBMARINE_HADOOP_VERSION=2.9

# Path to the submarine jars.
SUBMARINE_PATH=/opt/submarine-current

# Similar to HADOOP_CONF_DIR, location of the Hadoop configuration directory
HADOOP_CONF_PATH=/usr/local/hadoop/etc/hadoop

# Path to the MNIST example.
MNIST_PATH=/home/yarn/submarine

while [ $# -gt 0 ]; do
  case "$1" in
    --debug*)
      DEBUG=$1
      shift
      ;;
    *)
      break
      ;;
  esac
done

DEBUG_PORT=8000
if [ "$DEBUG" ]; then
  JAVA_CMD="java -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=${DEBUG_PORT}"
else
  JAVA_CMD="java"
fi

while getopts 'd:c' OPT; do
  case $OPT in
    d)
      DATA_URL="$OPTARG";;
    c)
      USE_DOCKER=1;;
  esac
done
shift $(($OPTIND - 1))

if [[ -n "$DATA_URL" ]]; then
  WORKER_CMD="venv/bin/python mnist_distributed.py --steps 2 --data_dir /tmp/data --working_dir /tmp/mode --mnist_data_url ${DATA_URL}"
else
  WORKER_CMD="venv/bin/python mnist_distributed.py --steps 2 --data_dir /tmp/data --working_dir /tmp/mode"
fi

if [[ -n "$USE_DOCKER" ]]; then
  WORKER_CMD="/opt/$WORKER_CMD"
  # tony-mnist-tf-1.13.1:0.0.1 is built from the Dockerfile.tony.tf.mnist.tf_1.13.1 under docs/helper/docker/tensorflow/mnist
  DOCKER_CONF="--conf tony.docker.containers.image=tony-mnist-tf-1.13.1:0.0.1 --conf tony.docker.enabled=true"
else
  WORKER_CMD="myvenv.zip/$WORKER_CMD"
fi

${JAVA_CMD} -cp $(${HADOOP_HOME}/bin/hadoop classpath --glob):${SUBMARINE_PATH}/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar:${HADOOP_CONF_PATH} \
 org.apache.submarine.client.cli.Cli job run --name tf-job-001 \
 --framework tensorflow \
 --verbose \
 --input_path "" \
 --num_workers 2 \
 --worker_resources memory=1G,vcores=1 \
 --num_ps 1 \
 --ps_resources memory=1G,vcores=1 \
 --worker_launch_cmd "${WORKER_CMD}" \
 --ps_launch_cmd "myvenv.zip/venv/bin/python mnist_distributed.py --steps 2 --data_dir /tmp/data --working_dir /tmp/mode" \
 --insecure \
 --conf tony.containers.resources=${MNIST_PATH}/myvenv.zip#archive,${MNIST_PATH}/mnist_distributed.py,${SUBMARINE_PATH}/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar \
 $DOCKER_CONF

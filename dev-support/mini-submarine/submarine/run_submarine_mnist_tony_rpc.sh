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

while getopts 'd:' OPT; do
  case $OPT in
    d)
      DATA_URL="$OPTARG";;
  esac
done
shift $(($OPTIND - 1))

if [[ -n "$DATA_URL" ]]; then
  WORKER_CMD="myvenv.zip/venv/bin/python mnist_distributed.py --steps 2 --data_dir /tmp/data --working_dir /tmp/mode --mnist_data_url ${DATA_URL}"
else
  WORKER_CMD="myvenv.zip/venv/bin/python mnist_distributed.py --steps 2 --data_dir /tmp/data --working_dir /tmp/mode"
fi 

SUBMARINE_VERSION=${SUBMARINE_VER:-"0.4.0-SNAPSHOT"}

HADOOP_VERSION=2.9

${JAVA_CMD} -cp /opt/submarine-dist-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}/submarine-all-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}.jar:/usr/local/hadoop/etc/hadoop:/opt/submarine-dist-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}/conf \
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
 --conf tony.containers.resources=/home/yarn/submarine/myvenv.zip#archive,/home/yarn/submarine/mnist_distributed.py,/opt/submarine-dist-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}/submarine-all-${SUBMARINE_VERSION}-hadoop-${HADOOP_VERSION}.jar

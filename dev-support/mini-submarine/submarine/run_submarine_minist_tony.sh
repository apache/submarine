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
SUBMARINE_VERSION=${SUBMARINE_VER}-hadoop-3.1
SUBMARINE_HOME=/opt/hadoop-submarine-dist-${SUBMARINE_VERSION}
CLASSPATH=$(hadoop classpath --glob):${SUBMARINE_HOME}/hadoop-submarine-core-${SUBMARINE_VER}.jar:${SUBMARINE_HOME}/hadoop-submarine-tony-runtime-${SUBMARINE_VER}.jar:${SUBMARINE_HOME}/tony-cli-0.3.18-all.jar \
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
 --conf tony.containers.resources=/home/yarn/submarine/myvenv.zip#archive,/home/yarn/submarine/mnist_distributed.py,${SUBMARINE_HOME}/tony-cli-0.3.18-all.jar

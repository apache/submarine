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


export JAVA_HOME=${JAVA_HOME:-$HOME/workspace/app/java}
export HADOOP_HOME=${HADOOP_HOME:-$HADOOP_HDFS_HOME}
export CLASSPATH=${CLASSPATH:-`hdfs classpath --glob`}
export ARROW_LIBHDFS_DIR=${ARROW_LIBHDFS_DIR:-$HADOOP_HOME/lib/native}

# path to pysubmarine/submarine
PYTHONPATH=$HOME/workspace/submarine/submarine-sdk/pysubmarine

HADOOP_CONF_PATH=${HADOOP_CONF_PATH:-$HADOOP_CONF_DIR}

SUBMARINE_VERSION=0.5.0
SUBMARINE_HADOOP_VERSION=2.9
SUBMARINE_JAR=/opt/submarine-dist-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}/submarine-dist-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}/submarine-all-${SUBMARINE_VERSION}-hadoop-${SUBMARINE_HADOOP_VERSION}.jar

java -cp $(${HADOOP_COMMON_HOME}/bin/hadoop classpath --glob):${SUBMARINE_JAR}:${HADOOP_CONF_PATH} \
 org.apache.submarine.client.cli.Cli job run --name afm-job-001 \
 --framework pytorch \
 --verbose \
 --input_path "" \
 --num_workers 2 \
 --worker_resources memory=1G,vcores=1 \
 --worker_launch_cmd "JAVA_HOME=$JAVA_HOME HADOOP_HOME=$HADOOP_HOME CLASSPATH=$CLASSPATH ARROW_LIBHDFS_DIR=$ARROW_LIBHDFS_DIR PYTHONPATH=$PYTHONPATH sdk.zip/sdk/bin/python run_afm.py --conf ./afm.json --task_type train" \
 --insecure \
 --conf tony.containers.resources=sdk.zip#archive,${SUBMARINE_JAR},run_afm.py,afm.json


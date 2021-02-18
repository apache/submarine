#!/usr/bin/env bash
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
set -euo pipefail
hadoop_v=2.9.2
spark_v=2.3.4

submarine_v=${submarine_version:-"0.6.0-SNAPSHOT"}
echo "Using submarine version: $submarine_v"

image_name="local/mini-submarine:${submarine_v}"

if [ -L ${BASH_SOURCE-$0} ]; then
  PWD=$(dirname $(readlink "${BASH_SOURCE-$0}"))
else
  PWD=$(dirname ${BASH_SOURCE-$0})
fi
export MINI_PATH=$(cd "${PWD}">/dev/null; pwd)
SUBMARINE_PROJECT_PATH=${MINI_PATH}/../..

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

is_empty_dir(){
  return `ls -A $1|wc -w`
}

# download hadoop
download_package "hadoop-${hadoop_v}.tar.gz" "http://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-${hadoop_v}"
# download spark
download_package "spark-${spark_v}-bin-hadoop2.7.tgz" "http://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-${spark_v}"
# download zookeeper
download_package "zookeeper-3.4.14.tar.gz" "http://mirror.bit.edu.cn/apache/zookeeper/zookeeper-3.4.14"


if [ ! -d "${SUBMARINE_PROJECT_PATH}/submarine-dist/target" ]; then
  mkdir "${SUBMARINE_PROJECT_PATH}/submarine-dist/target"
fi

submarine_dist_exists=$(find -L "${SUBMARINE_PROJECT_PATH}/submarine-dist/target" -name "submarine-dist-${submarine_v}*.tar.gz")

# If exists, use the release candidate artifacts to build image
if [[ ! -z "${release_candidates_path:-}" ]]; then
  submarine_dist_exists=${release_candidates_path}
  echo "Using release candidates artifacts: ${release_candidates_path}"
  cp ${release_candidates_path}/submarine-dist-${submarine_v}-hadoop*.tar.gz ${MINI_PATH}
fi

# Build source code if the package doesn't exist.
if [[ -z "${submarine_dist_exists}" ]]; then
  cd "${SUBMARINE_PROJECT_PATH}"
  mvn clean package -DskipTests
fi

cp ${SUBMARINE_PROJECT_PATH}/submarine-dist/target/submarine-dist-${submarine_v}*.tar.gz ${MINI_PATH}
cp -r ${SUBMARINE_PROJECT_PATH}/submarine-sdk/pysubmarine ${MINI_PATH}
cp -r ${SUBMARINE_PROJECT_PATH}/dev-support/database ${MINI_PATH}

# build image
echo "Start building the mini-submarine docker image..."
cd ${MINI_PATH}
docker build --build-arg HADOOP_VERSION=${hadoop_v} --build-arg SPARK_VERSION=${spark_v} --build-arg SUBMARINE_VERSION=${submarine_v} --build-arg IMAGE_NAME=${image_name} -t ${image_name} .

# clean template file
rm -rf ${MINI_PATH}/database
rm -rf ${MINI_PATH}/pysubmarine
rm -rf ${MINI_PATH}/submarine-dist-${submarine_v}*.tar.gz

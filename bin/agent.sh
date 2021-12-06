#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

USAGE="Usage: bin/agent.sh [--config <conf-dir>]"

if [[ "$1" == "--config" ]]; then
  shift
  conf_dir="$1"
  if [[ ! -d "${conf_dir}" ]]; then
    echo "ERROR : ${conf_dir} is not a directory"
    echo ${USAGE}
    exit 1
  else
    export SUBMARINE_CONF_DIR="${conf_dir}"
  fi
  shift
fi

if [ -L ${BASH_SOURCE-$0} ]; then
  BIN=$(dirname $(readlink "${BASH_SOURCE-$0}"))
else
  BIN=$(dirname ${BASH_SOURCE-$0})
fi
export BIN=$(cd "${BIN}">/dev/null; pwd)
GET_MYSQL_JAR=false

. "${BIN}/common.sh"

cd ${BIN}/>/dev/null

SUBMARINE_AGENT_NAME="Submarine Agent"
SUBMARINE_AGENT_LOGFILE="${SUBMARINE_LOG_DIR}/agent.log"
SUBMARINE_AGENT_MAIN=org.apache.submarine.server.k8s.agent.SubmarineAgent
JAVA_OPTS+="${SUBMARINE_APP_JAVA_OPTS:-""} ${SUBMARINE_APP_MEM:-""} -Dsubmarine.log.file=${SUBMARINE_AGENT_LOGFILE}"

add_jar_in_dir "${BIN}/../lib"

if [[ ! -d "${SUBMARINE_LOG_DIR}" ]]; then
  echo "Log dir doesn't exist, create ${SUBMARINE_LOG_DIR}"
  $(mkdir -p "${SUBMARINE_LOG_DIR}")
fi

exec $JAVA_RUNNER $JAVA_OPTS -cp ${SUBMARINE_APP_CLASSPATH} ${SUBMARINE_AGENT_MAIN} "$@" | tee -a "${SUBMARINE_AGENT_LOGFILE}" 2>&1

#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if [[ -L ${BASH_SOURCE-$0} ]]; then
  FWDIR=$(dirname $(readlink "${BASH_SOURCE-$0}"))
else
  FWDIR=$(dirname "${BASH_SOURCE-$0}")
fi

if [[ -z "${SUBMARINE_HOME}" ]]; then
  # Make SUBMARINE_HOME look cleaner in logs by getting rid of the
  # extra ../
  export SUBMARINE_HOME="$(cd "${FWDIR}/.."; pwd)"
fi

if [[ -z "${SUBMARINE_CONF_DIR}" ]]; then
  export SUBMARINE_CONF_DIR="${SUBMARINE_HOME}/conf"
fi

if [[ -z "${SUBMARINE_LOG_DIR}" ]]; then
  export SUBMARINE_LOG_DIR="${SUBMARINE_HOME}/logs"
fi

if [[ -f "${SUBMARINE_CONF_DIR}/submarine-env.sh" ]]; then
  . "${SUBMARINE_CONF_DIR}/submarine-env.sh"
fi

WORKBENCH_CLASSPATH+=":${SUBMARINE_CONF_DIR}"

function add_each_jar_in_dir(){
  if [[ -d "${1}" ]]; then
    for jar in $(find -L "${1}" -maxdepth 1 -name '*jar'); do
      WORKBENCH_CLASSPATH="$jar:$WORKBENCH_CLASSPATH"
    done
  fi
}

function add_each_jar_in_dir_recursive(){
  if [[ -d "${1}" ]]; then
    for jar in $(find -L "${1}" -type f -name '*jar'); do
      WORKBENCH_CLASSPATH="$jar:$WORKBENCH_CLASSPATH"
    done
  fi
}

function add_jar_in_dir(){
  if [[ -d "${1}" ]]; then
    WORKBENCH_CLASSPATH="${1}/*:${WORKBENCH_CLASSPATH}"
  fi
}

JAVA_OPTS+=" ${WORKBENCH_JAVA_OPTS} -Dfile.encoding=UTF-8 ${WORKBENCH_MEM}"
JAVA_OPTS+=" -Dlog4j.configuration=file://${SUBMARINE_CONF_DIR}/log4j.properties"
export JAVA_OPTS

if [[ -n "${JAVA_HOME}" ]]; then
  JAVA_RUNNER="${JAVA_HOME}/bin/java"
else
  JAVA_RUNNER=java
fi
export JAVA_RUNNER

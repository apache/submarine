#!/bin/bash
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
#
# description: Start and stop daemon script for.
#

set -euo pipefail
set -x

USAGE="-e Usage: submarine-daemon.sh {start|stop|restart|status}"

if [ -L ${BASH_SOURCE-$0} ]; then
  BIN=$(dirname $(readlink "${BASH_SOURCE-$0}"))
else
  BIN=$(dirname ${BASH_SOURCE-$0})
fi
export BIN=$(cd "${BIN}">/dev/null; pwd)
GET_MYSQL_JAR=false

. "${BIN}/common.sh"

cd ${BIN}/>/dev/null

SUBMARINE_SERVER_NAME="Submarine Server"
SUBMARINE_SERVER_LOGFILE="${SUBMARINE_LOG_DIR}/submarine.log"
SUBMARINE_SERVER_MAIN=org.apache.submarine.server.SubmarineServer
JAVA_OPTS+="${SUBMARINE_SERVER_JAVA_OPTS:-""} ${SUBMARINE_SERVER_MEM:-""} -Dsubmarine.log.file=${SUBMARINE_SERVER_LOGFILE}"

add_jar_in_dir "${BIN}/../lib"
add_jar_in_dir "${BIN}/../lib/submitter"

function initialize_default_directories() {
  if [[ ! -d "${SUBMARINE_LOG_DIR}" ]]; then
    echo "Log dir doesn't exist, create ${SUBMARINE_LOG_DIR}"
    $(mkdir -p "${SUBMARINE_LOG_DIR}")
  fi
}

function found_submarine_server_pid() {
  process='SubmarineServer';
  RUNNING_PIDS=$(ps x | grep ${process} | grep -v grep | awk '{print $1}');

  if [[ -z "${RUNNING_PIDS}" ]]; then
    return
  fi

  if ! kill -0 ${RUNNING_PIDS} > /dev/null 2>&1; then
    echo "${SUBMARINE_SERVER_NAME} running but process is dead"
  fi

  echo "${RUNNING_PIDS}"
}

function wait_for_submarine_server_to_die() {
  local pid
  local count

  pid=`found_submarine_server_pid`
  timeout=10
  count=0
  timeoutTime=$(date "+%s")
  let "timeoutTime+=$timeout"
  currentTime=$(date "+%s")
  forceKill=1

  while [[ $currentTime -lt $timeoutTime ]]; do
    $(kill ${pid} > /dev/null 2> /dev/null)
    if kill -0 ${pid} > /dev/null 2>&1; then
      sleep 3
    else
      forceKill=0
      break
    fi
    currentTime=$(date "+%s")
  done

  if [[ forceKill -ne 0 ]]; then
    $(kill -9 ${pid} > /dev/null 2> /dev/null)
  fi
}

function check_jdbc_jar() {
  if [[ -d "${1}" ]]; then
    mysql_connector_exists=$(find -L "${1}" -name "mysql-connector*")
    if [[ -z "${mysql_connector_exists}" ]]; then
      if [[ ${GET_MYSQL_JAR} = true ]]; then
        download_mysql_jdbc_jar
      else
        echo -e "\\033[31mError: There is no mysql jdbc jar in lib.\\033[0m"
        echo -e "\\033[31mPlease download a mysql jdbc jar and put it under lib manually.\\033[0m"
        echo -e "\\033[31mOr add a parameter getMysqlJar, like this:\n./bin/submarine-daemon.sh start getMysqlJar\\033[0m"
        echo -e "\\033[31mIt would download mysql jdbc jar automatically.\\033[0m"
        exit 1
      fi
    fi
  fi
}

function start() {
  local pid

  pid=`found_submarine_server_pid`
  if [[ ! -z "$pid" && "$pid" != 0 ]]; then
    echo "${SUBMARINE_SERVER_NAME}:${pid} is already running"
    return 0;
  fi

  check_jdbc_jar "${BIN}/../lib"

  initialize_default_directories

  echo "SUBMARINE_SERVER_CLASSPATH: ${SUBMARINE_SERVER_CLASSPATH}" >> "${SUBMARINE_SERVER_LOGFILE}"

  nohup $JAVA_RUNNER $JAVA_OPTS -cp $SUBMARINE_SERVER_CLASSPATH $SUBMARINE_SERVER_MAIN >> "${SUBMARINE_SERVER_LOGFILE}" 2>&1 < /dev/null &
  pid=$!
  if [[ ! -z "${pid}" ]]; then
    echo "${SUBMARINE_SERVER_NAME} start"
    return 1;
  fi
}

function stop() {
  local pid
  pid=`found_submarine_server_pid`

  if [[ -z "$pid" ]]; then
    echo "${SUBMARINE_SERVER_NAME} is not running"
    return 0;
  else
    # submarine workbench daemon kill
    wait_for_submarine_server_to_die
    echo "${SUBMARINE_SERVER_NAME} stop"
  fi
}

function find_submarine_server_process() {
  local pid
  pid=`found_submarine_server_pid`

  if [[ -z "$pid" ]]; then
    echo "${SUBMARINE_SERVER_NAME} is not running"
    return 1
  else
    if ! kill -0 ${pid} > /dev/null 2>&1; then
      echo "${SUBMARINE_SERVER_NAME} running but process is dead"
      return 1
    else
      echo "${SUBMARINE_SERVER_NAME} is running"
    fi
  fi
}

if [[ "${2:-""}" = "getMysqlJar" ]]; then
  export GET_MYSQL_JAR=true
fi

case "${1}" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    echo "${SUBMARINE_SERVER_NAME} is restarting" >> "${SUBMARINE_SERVER_LOGFILE}"
    stop
    start
    ;;
  status)
    find_submarine_server_process
    ;;
  *)
    echo ${USAGE}
esac

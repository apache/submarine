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

USAGE="-e Usage: workbench-daemon.sh {start|stop|restart|reload|status}"

if [ -L ${BASH_SOURCE-$0} ]; then
  BIN=$(dirname $(readlink "${BASH_SOURCE-$0}"))
else
  BIN=$(dirname ${BASH_SOURCE-$0})
fi
BIN=$(cd "${BIN}">/dev/null; pwd)

. "${BIN}/common.sh"

WORKBENCH_NAME="Submarine Workbench"
WORKBENCH_LOGFILE="${SUBMARINE_LOG_DIR}/workbench.log"
WORKBENCH_MAIN=org.apache.submarine.server.WorkbenchServer
JAVA_OPTS+=" -Dworkbench.log.file=${WORKBENCH_LOGFILE}"

addJarInDir "${BIN}/../workbench"
addJarInDir "${BIN}/../workbench/lib"

function initialize_default_directories() {
  if [[ ! -d "${SUBMARINE_LOG_DIR}" ]]; then
    echo "Log dir doesn't exist, create ${SUBMARINE_LOG_DIR}"
    $(mkdir -p "${SUBMARINE_LOG_DIR}")
  fi
}

function foundWorkbenchServerPid() {
  process=WorkbenchServer;
  RUNNING_PIDS=$(ps x | grep $process | grep -v grep | awk '{print $1}');

  if [[ -z "${RUNNING_PIDS}" ]]; then
    return
  fi

  if ! kill -0 ${RUNNING_PIDS} > /dev/null 2>&1; then
    echo "${WORKBENCH_NAME} running but process is dead"
  fi

  echo "${RUNNING_PIDS}"
}

function wait_for_workbench_to_die() {
  local pid
  local count

  pid=`foundWorkbenchServerPid`
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

function start() {
  local pid

  pid=`foundWorkbenchServerPid`
  if [[ ! -z "$pid" && "$pid" != 0 ]]; then
    echo "${WORKBENCH_NAME}:${pid} is already running"
    return 0;
  fi

  initialize_default_directories

  echo "WORKBENCH_CLASSPATH: ${WORKBENCH_CLASSPATH}" >> "${WORKBENCH_LOGFILE}"

  nohup $JAVA_RUNNER $JAVA_OPTS -cp $WORKBENCH_CLASSPATH $WORKBENCH_MAIN >> "${WORKBENCH_LOGFILE}" 2>&1 < /dev/null &
  pid=$!
  if [[ ! -z "${pid}" ]]; then
    echo "${WORKBENCH_NAME} start"
    return 1;
  fi
}

function stop() {
  local pid
  pid=`foundWorkbenchServerPid`

  if [[ -z "$pid" ]]; then
    echo "${WORKBENCH_NAME} is not running"
    return 0;
  else
    # submarine workbench daemon kill
    wait_for_workbench_to_die
    echo "${WORKBENCH_NAME} stop"
  fi
}

function find_workbench_process() {
  local pid
  pid=`foundWorkbenchServerPid`

  if [[ -z "$pid" ]]; then
    if ! kill -0 ${pid} > /dev/null 2>&1; then
      echo "${WORKBENCH_NAME} running but process is dead"
      return 1
    else
      echo "${WORKBENCH_NAME} is running"
    fi
  else
    echo "${WORKBENCH_NAME} is not running"
    return 1
  fi
}

case "${1}" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  reload)
    stop
    start
    ;;
  restart)
    echo "${WORKBENCH_NAME} is restarting" >> "${WORKBENCH_LOGFILE}"
    stop
    start
    ;;
  status)
    find_workbench_process
    ;;
  *)
    echo ${USAGE}
esac

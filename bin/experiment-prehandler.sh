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

export SUBMARINE_HOME="$(cd "${FWDIR}/.."; pwd)"
export SUBMARINE_LOG_DIR="${SUBMARINE_HOME}/logs"

if [[ ! -d "${SUBMARINE_LOG_DIR}" ]]; then
  echo "Log dir doesn't exist, create ${SUBMARINE_LOG_DIR}"
  $(mkdir -p "${SUBMARINE_LOG_DIR}")
fi

export CLASSPATH=`$HADOOP_HOME/bin/hadoop classpath --glob`

python3 /opt/submarine-experiment-prehandler/prehandler_main.py

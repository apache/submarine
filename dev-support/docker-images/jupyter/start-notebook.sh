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
#
# description: Start and stop daemon script for.
#

set -euo pipefail

# Install conda dependency
if [[ -n "${INSTALL_ENVIRONMENT_COMMAND:-}" ]]; then
  /bin/bash -c "${INSTALL_ENVIRONMENT_COMMAND}"
fi

NOTEBOOK_ARGS="--ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'"
NB_USER="${NB_USER:-"jovyan"}"
NB_PREFIX="${NB_PREFIX:-"/"}"
NB_PORT="${NB_PORT:-8888}"

if [[ -n "${NB_USER}" ]]; then
  NOTEBOOK_ARGS="--notebook-dir=/home/${NB_USER} ${NOTEBOOK_ARGS}"
fi

if [[ -n "${NB_PORT}" ]]; then
  NOTEBOOK_ARGS="--port=${NB_PORT} ${NOTEBOOK_ARGS}"
fi

if [[ -n "${NB_PREFIX}" ]]; then
  NOTEBOOK_ARGS="--NotebookApp.base_url=${NB_PREFIX} ${NOTEBOOK_ARGS}"
fi

/bin/bash -c "jupyter notebook ${NOTEBOOK_ARGS}"

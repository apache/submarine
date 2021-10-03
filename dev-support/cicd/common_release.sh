#!/usr/bin/env bash

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

# common functions

if [[ -z "${TAR:-}" ]]; then
  TAR="/usr/bin/tar"
fi

if [[ -z "${SHASUM:-}" ]]; then
  SHASUM="/usr/bin/shasum"
fi

if [[ -z "${WORKING_DIR:-}" ]]; then
  WORKING_DIR="/tmp/submarine-release"
fi

DEBUG_SUBMARINE_SCRIPT=false
if $DEBUG_SUBMARINE_SCRIPT; then
  echo "DEBUGGING, skip remove ${WORKING_DIR}"
else
  echo "Cleaning up ${WORKING_DIR}"
  rm -rf "${WORKING_DIR}"
  mkdir "${WORKING_DIR}"
fi

# If set to 'yes', release script will deploy artifacts to SNAPSHOT repository.
DO_SNAPSHOT='no'

usage() {
  echo "usage) $0 [Release version] [Branch or Tag]"
  echo "   ex. $0 0.7.0 v0.7.0"
  exit 1
}

function git_clone() {
  echo "Clone the source"
  # clone source
  git clone https://git-wip-us.apache.org/repos/asf/submarine.git "${WORKING_DIR}/submarine"

  if [[ $? -ne 0 ]]; then
    echo "Can not clone source repository"
    exit 1
  fi

  cd "${WORKING_DIR}/submarine"
  git checkout "${GIT_TAG}"
  echo "Checked out ${GIT_TAG}"
}

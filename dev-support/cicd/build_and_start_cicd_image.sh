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


if [ -L ${BASH_SOURCE-$0} ]; then
  PWD=$(dirname $(readlink "${BASH_SOURCE-$0}"))
else
  PWD=$(dirname ${BASH_SOURCE-$0})
fi
CURRENT_PATH=$(cd "${PWD}">/dev/null; pwd)

# So it can executed from any directory
# e.g. We can run ./dev-support/cicd/build_and_start_cicd_image.sh from the root of submarine project
cd ${CURRENT_PATH}

printf "Building Submarine CI/CD Image.\n"
docker build -t submarine-cicd .
printf "Start Submarine CI/CD.\n"
docker run -it --rm -p 4000:4000 submarine-cicd

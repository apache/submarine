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

# Description: Run a frontend E2E tests
# Usage: ./run_frontend_e2e.sh [Testcase]
#       Testcase: Check the directory "submarine-test/test-e2e/src/test/java/org/apache/submarine/integration"
#       Example : ./run_frontend_e2e.sh loginIT

# ======= Modifiable Variables ======= #
# Note: URL must start with "http" 
# (Ref: https://www.selenium.dev/selenium/docs/api/java/org/openqa/selenium/WebDriver.html#get(java.lang.String))
WORKBENCH_PORT=4200
URL="http://127.0.0.1"
# ==================================== #

HTTP_CODE=$(curl -sL -w "%{http_code}\\n" $URL:$WORKBENCH_PORT -o /dev/null)
if [[ "$HTTP_CODE" != "200" ]]; then
  echo "Make sure Submarine Workbench is running on $URL:$WORKBENCH_PORT"
  exit 1
else
  echo "HTTP_CODE ($URL:$WORKBENCH_PORT): $HTTP_CODE"
fi

set -e
TESTCASE=$1
mvn -DSUBMARINE_WORKBENCH_URL=$URL -DSUBMARINE_WORKBENCH_PORT=$WORKBENCH_PORT -Dtest=$TESTCASE -DSUBMARINE_E2E_LOCAL=true test

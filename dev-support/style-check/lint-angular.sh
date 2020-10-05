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

npm install prettier@^2.0.5

WORKBENCH_NG=./submarine-workbench/workbench-web

PRETTIER_ERRORS=$(./node_modules/.bin/prettier --check --trailing-comma none "$WORKBENCH_NG/src/**/*.{ts,html}" | grep "Forgot to run Prettier?")


if test "$PRETTIER_ERRORS"; then
    echo -e "prettier checks failed at following occurrences:\n$PRETTIER_ERRORS\n"
    echo -e "Please use \\033[31m"./node_modules/.bin/prettier --write --trailing-comma none "$WORKBENCH_NG/src/**/*.{ts,html}""\\033[0m to format code automatically\n"
    exit 1
else
    echo -e "Checkstyle checks passed."
fi


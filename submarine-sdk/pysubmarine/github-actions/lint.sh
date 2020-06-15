#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"
cd ..

pycodestyle --max-line-length=100  -- submarine tests
pylint --ignore job --msg-template="{path} ({line},{column}): [{msg_id} {symbol}] {msg}" --rcfile=pylintrc -- submarine tests
./github-actions/auto-format.sh

GIT_STATUS="$(git status --porcelain)"
GIT_DIFF="$(git diff)"
if [ "$GIT_STATUS" ]; then
	echo "Code is not formatted by yapf and isort. Please run ./github-actions/auto-format.sh"
	echo "Git status is"
	echo "------------------------------------------------------------------"
	echo "$GIT_STATUS"
	echo "Git diff is"
	echo "------------------------------------------------------------------"
	echo "$GIT_DIFF"
	exit 1
else
	echo "Test successful"
fi

set +ex

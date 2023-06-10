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

set -euxo pipefail

FWDIR="$(cd "$(dirname "$0")"; pwd)"
cd "$FWDIR"
cd ../../../

# Sort imports
isort submarine-sdk/ dev-support/ website/
# Replace `'long': int if six.PY3 else long,  # noqa: F821` to `'long': int,`
sed -i '' 's/ if six.PY3 else long,  # noqa: F821/,/g' ./submarine-sdk/pysubmarine/submarine/client/api_client.py
# Autoformat code
black submarine-sdk/ dev-support/ website/ --preview 

set +euxo pipefail

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

set -euo pipefail
set -x

wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O "$HOME"/anaconda.sh;
bash "$HOME"/anaconda.sh -b -p /usr/bin/anaconda
export PATH="/usr/bin/anaconda/bin:$PATH"
cd "$HOME"
# Useful for debugging any issues with conda
conda info -a
if [[ -n "${PYTHON_VERSION:-}" ]]; then
  conda create -q -n submarine-dev python="$PYTHON_VERSION"
else
  conda create -q -n submarine-dev python=3.6
fi

source activate submarine-dev
python --version
pip install --upgrade pip

# Install pysubmarine
git clone https://github.com/apache/submarine.git
cd submarine/submarine-sdk/pysubmarine
pip install -e .[tf,pytorch]
pip install -r github-actions/lint-requirements.txt
pip install -r github-actions/test-requirements.txt

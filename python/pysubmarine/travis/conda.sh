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

#!/usr/bin/env bash

set -ex
sudo mkdir -p /travis-install
sudo chown travis /travis-install

# We do this conditionally because it saves us some downloading if the
# version is the same.
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O /travis-install/miniconda.sh;
else
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /travis-install/miniconda.sh;
fi

bash /travis-install/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
# Useful for debugging any issues with conda
conda info -a
if [[ -n "$TRAVIS_PYTHON_VERSION" ]]; then
  conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION
else
  conda create -q -n test-environment python=3.6
fi
source activate test-environment
python --version
pip install --upgrade pip
pip install -r ./python/pysubmarine/travis/test-requirements.txt

pip install ./python/pysubmarine/.
export SUBMARINE_HOME=$(pwd)

# Print current environment info
pip list
echo $SUBMARINE_HOME

# Turn off trace output & exit-on-errors
set +ex

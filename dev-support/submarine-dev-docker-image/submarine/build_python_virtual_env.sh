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

#!/bin/bash
wget https://files.pythonhosted.org/packages/33/bc/fa0b5347139cd9564f0d44ebd2b147ac97c36b2403943dbee8a25fd74012/virtualenv-16.0.0.tar.gz
tar xf virtualenv-16.0.0.tar.gz

wget https://github.com/linkedin/TonY/releases/download/v0.3.18/tony-cli-0.3.18-all.jar
mv tony-cli-0.3.18-all.jar /opt/hadoop-submarine-dist-${SUBMARINE_VER}-hadoop-3.1/tony-cli-0.3.18-all.jar
# Make sure to install using Python 3, as TensorFlow only provides Python 3 artifacts
python3 virtualenv-16.0.0/virtualenv.py venv
. venv/bin/activate
pip install tensorflow==1.13.1
zip -r myvenv.zip venv
deactivate


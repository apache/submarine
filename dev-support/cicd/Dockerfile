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

FROM ubuntu:18.04
RUN \
  apt-get update && \
  apt-get install -y wget vim git python-pip zlib1g-dev libssl-dev && \
  wget https://www.python.org/ftp/python/2.7.13/Python-2.7.13.tgz && \
  tar xzf Python-2.7.13.tgz && cd Python-2.7.13 && ./configure --with-zlib && make install && make altinstall && \
  cd /root && wget https://files.pythonhosted.org/packages/59/38/55dd25a965990bd93f77eb765b189e72cf581ce1c2de651cb7b1dea74ed1/virtualenv-16.2.0.tar.gz && \
  tar xf virtualenv-16.2.0.tar.gz && \
  python virtualenv-16.2.0/virtualenv.py venv2.7 && \
  . venv2.7/bin/activate && \
  pip install jira && \
  deactivate

RUN \
  git clone https://gitbox.apache.org/repos/asf/submarine.git && \
  cd submarine && \
  git remote rename origin apache && \
  git remote add apache-github https://github.com/apache/submarine.git
ADD \
  entry.sh /entry.sh
ENV \
  SUBMARINE_HOME=/submarine \
  PYTHON_VENV_PATH=/root

CMD /entry.sh

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

FROM ubuntu:16.04
RUN \
  apt-get update && \
  apt-get install -y vim python git python-pip && \
  pip install jira
RUN \
  git clone https://gitbox.apache.org/repos/asf/hadoop-submarine.git && \
  cd hadoop-submarine && \
  git remote rename origin apache && \
  git remote add apache-github https://github.com/apache/hadoop-submarine.git
ADD \
  entry.sh /entry.sh
ENV \
  SUBMARINE_HOME=/hadoop-submarine
CMD /entry.sh

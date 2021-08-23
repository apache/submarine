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

FROM python:3.7
MAINTAINER Apache Software Foundation <dev@submarine.apache.org>

ADD ./tmp/submarine-sdk /opt/
RUN pip install /opt/pysubmarine
RUN pip install tensorflow==2.3.0
RUN pip install tensorboard
RUN pip install tensorflow_datasets==2.1.0

ADD ./mnist_keras_distributed.py /opt/
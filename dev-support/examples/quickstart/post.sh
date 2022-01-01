#!/usr/bin/env bash
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

curl -X POST -H "Content-Type: application/json" -d '
{
  "meta": {
    "name": "quickstart",
    "namespace": "default",
    "framework": "TensorFlow",
    "cmd": "python /opt/train.py",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "environment": {
    "image": "apache/submarine:quickstart-0.7.0-SNAPSHOT"
  },
  "spec": {
    "Worker": {
      "replicas": 3,
      "resources": "cpu=1,memory=1024M"
    }
  }
}
' http://127.0.0.1:32080/api/v1/experiment

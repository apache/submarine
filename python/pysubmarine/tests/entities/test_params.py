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

from submarine.entities import Param


def _check(param, key, value, worker_index):
    assert type(param) == Param
    assert param.key == key
    assert param.value == value
    assert param.worker_index == worker_index


def test_creation_and_hydration():
    key = "alpha"
    value = 10000
    worker_index = 1

    param = Param(key, value, worker_index)
    _check(param, key, value, worker_index)

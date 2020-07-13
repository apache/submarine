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

import json
from os import environ

import pytest

from submarine.utils.env import (get_env, get_from_dicts, get_from_json,
                                 get_from_registry, unset_variable)


@pytest.fixture(scope="function")
def output_json_filepath():
    params = {"learning_rate": 0.05}
    path = '/tmp/data.json'
    with open(path, 'w') as f:
        json.dump(params, f)
    return path


def test_get_env():
    environ["test"] = "hello"
    assert get_env("test") == "hello"


def test_unset_variable():
    environ["test"] = "hello"
    unset_variable("test")
    assert "test" not in environ


def test_merge_json(output_json_filepath):
    default_params = {"learning_rate": 0.08, "embedding_size": 256}
    params = get_from_json(output_json_filepath, default_params)
    assert params['learning_rate'] == 0.05
    assert params['embedding_size'] == 256


def test_merge_dicts():
    params = {"learning_rate": 0.05}
    default_params = {"learning_rate": 0.08, "embedding_size": 256}
    final = get_from_dicts(params, default_params)
    assert final['learning_rate'] == 0.05
    assert final['embedding_size'] == 256


def test_get_from_registry():
    registry = {'model': 'xgboost'}
    val = get_from_registry('MODEL', registry)
    assert val == 'xgboost'

    with pytest.raises(ValueError):
        get_from_registry('test', registry)

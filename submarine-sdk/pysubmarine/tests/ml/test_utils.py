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

import pytest
import json
from submarine.ml.utils import sanity_checks, merge_json, merge_dicts,\
    get_from_registry, get_TFConfig


@pytest.fixture(scope="function")
def output_json_filepath():
    params = {"learning_rate": 0.05}
    path = '/tmp/data.json'
    with open(path, 'w') as f:
        json.dump(params, f)
    return path


def test_sanity_checks():
    params = {"learning rate": 0.05}
    with pytest.raises(AssertionError, match="Does not define any input parameters"):
        sanity_checks(params)

    params.update({'input': {'train_data': '/tmp/train.csv'}})
    with pytest.raises(AssertionError, match="Does not define any input type"):
        sanity_checks(params)

    params.update({'input': {'type': 'libsvm'}})
    with pytest.raises(AssertionError, match="Does not define any output parameters"):
        sanity_checks(params)

    params.update({'output': {'save_model_dir': '/tmp/save'}})


def test_merge_json(output_json_filepath):
    defaultParams = {"learning_rate": 0.08, "embedding_size": 256}
    params = merge_json(output_json_filepath, defaultParams)
    assert params['learning_rate'] == 0.05
    assert params['embedding_size'] == 256


def test_merge_dicts():
    params = {"learning_rate": 0.05}
    defaultParams = {"learning_rate": 0.08, "embedding_size": 256}
    final = merge_dicts(params, defaultParams)
    assert final['learning_rate'] == 0.05
    assert final['embedding_size'] == 256


def test_get_from_registry():
    registry = {'model': 'xgboost'}
    val = get_from_registry('MODEL', registry)
    assert val == 'xgboost'

    with pytest.raises(ValueError):
        get_from_registry('test', registry)


def test_get_TFConfig():
    params = {'training': {'mode': 'test'}}
    with pytest.raises(ValueError, match="mode should be local or distributed"):
        get_TFConfig(params)

    # run local training
    params.update({'training': {'mode': 'local', 'num_gpu': 0, 'num_threads': 4, 'log_steps': 10}})
    get_TFConfig(params)

    # run distributed training
    params.update({'training': {'mode': 'distributed', 'log_steps': 10}})
    get_TFConfig(params)

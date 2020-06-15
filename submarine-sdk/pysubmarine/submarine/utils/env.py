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

import copy
import json
import os
from collections import Mapping


def get_env(variable_name):
    return os.environ.get(variable_name)


def unset_variable(variable_name):
    if variable_name in os.environ:
        del os.environ[variable_name]


def check_env_exists(variable_name):
    if variable_name not in os.environ:
        return False
    return True


def get_from_json(path, defaultParams):
    """
    If model parameters not specify in Json, use parameter in defaultParams
    :param path: The json file that specifies the model parameters.
    :type path: String
    :param defaultParams: default parameters for model
    :type path: Dictionary
    :return:
    """
    if path is None or not os.path.isfile(path):
        return defaultParams
    with open(path) as json_data:
        params = json.load(json_data)
    return get_from_dicts(params, defaultParams)


def get_from_dicts(params, defaultParams):
    """
    If model parameters not specify in params, use parameter in defaultParams
    :param params: parameters which will be merged
    :type params: Dictionary
    :param defaultParams: default parameters for model
    :type params: Dictionary
    :return:
    """
    if params is None:
        return defaultParams

    dct = copy.deepcopy(defaultParams)
    for k, _ in params.items():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(defaultParams[k], Mapping)):
            dct[k] = get_from_dicts(params[k], dct[k])
        else:
            dct[k] = params[k]
    return dct


def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()
    if key in registry:
        return registry[key]
    else:
        raise ValueError('Key {} not supported, available options: {}'.format(
            key, registry.keys()))

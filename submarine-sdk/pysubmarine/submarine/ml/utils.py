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

import logging
import tensorflow as tf
import json
import os
import copy
from collections import Mapping

logger = logging.getLogger(__name__)


def sanity_checks(params):
    assert 'input' in params, (
        'Does not define any input parameters'
    )
    assert 'type' in params['input'], (
        'Does not define any input type'
    )
    assert 'output' in params, (
        'Does not define any output parameters'
    )


def merge_json(path, defaultParams):
    """
    Merge parameters json parameter into default parameters
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
    return merge_dicts(params, defaultParams)


def merge_dicts(params, defaultParams):
    """
    If model parameter not specify in params, use parameter in defaultParams
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
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(defaultParams[k], Mapping)):
            dct[k] = merge_dicts(params[k], dct[k])
        else:
            dct[k] = params[k]
    return dct


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance with appropriate device_filters set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if tf_config and 'task' in tf_config and 'type' in tf_config['task'] \
            and 'index' in tf_config['task']:
        # Master should only communicate with itself and ps.
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps.
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(  # gpu_options=gpu_options,
                                  device_filters=[
                                      '/job:ps',
                                      '/job:worker/task:%d' % tf_config['task'][
                                          'index']
                                  ])
    return None


def get_TFConfig(params):
    """
    Get TF_CONFIG to run local or distributed training, If user don't set TF_CONFIG environment
    variables, by default set local mode
    :param params: model parameters that contain total number of gpu or cpu the model
    intends to use
    :type params: Dictionary
    :return: The class specifies the configurations for an Estimator run
    """
    if params["training"]['mode'] == 'local':  # local mode
        tf_config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(
                device_count={'GPU': params["training"]['num_gpu'],
                              'CPU': params["training"]['num_threads']}),
            log_step_count_steps=params["training"]['log_steps'],
            save_summary_steps=params["training"]['log_steps'])

    elif params["training"]['mode'] == 'distributed':
        tf_config = tf.estimator.RunConfig(
            experimental_distribute=tf.contrib.distribute.DistributeConfig(
                train_distribute=tf.contrib.distribute.ParameterServerStrategy(),
                eval_distribute=tf.contrib.distribute.MirroredStrategy()),
            session_config=_get_session_config_from_env_var(),
            save_summary_steps=params["training"]['log_steps'],
            log_step_count_steps=params["training"]['log_steps'])
    else:
        raise ValueError("mode should be local or distributed")
    return tf_config


def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()
    if key in registry:
        return registry[key]
    else:
        raise ValueError(
            'Key {} not supported, available options: {}'.format(
                key, registry.keys()
            )
        )

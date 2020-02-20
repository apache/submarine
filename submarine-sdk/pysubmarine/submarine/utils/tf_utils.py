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

import tensorflow as tf
import json
import os


def _get_session_config_from_env_var(params):
    """Returns a tf.ConfigProto instance with appropriate device_filters set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if tf_config and 'task' in tf_config and 'type' in tf_config['task'] \
            and 'index' in tf_config['task']:
        # Master should only communicate with itself and ps.
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'],
                                  intra_op_parallelism_threads=params["resource"]['num_thread'],
                                  inter_op_parallelism_threads=params["resource"]['num_thread'])
        # Worker should only communicate with itself and ps.
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(  # gpu_options=gpu_options,
                device_filters=['/job:ps', '/job:worker/task:%d' % tf_config['task']['index']],
                intra_op_parallelism_threads=params["resource"]['num_thread'],
                inter_op_parallelism_threads=params["resource"]['num_thread'])
    return None


def get_tf_config(params):
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
                device_count={'GPU': params["resource"]['num_gpu'],
                              'CPU': params["resource"]['num_cpu']},
                intra_op_parallelism_threads=params["resource"]['num_thread'],
                inter_op_parallelism_threads=params["resource"]['num_thread']),
            log_step_count_steps=params["training"]['log_steps'],
            save_summary_steps=params["training"]['log_steps'])

    elif params["training"]['mode'] == 'distributed':
        tf_config = tf.estimator.RunConfig(
            experimental_distribute=tf.contrib.distribute.DistributeConfig(
                train_distribute=tf.contrib.distribute.ParameterServerStrategy(),
                eval_distribute=tf.contrib.distribute.ParameterServerStrategy()),
            session_config=_get_session_config_from_env_var(params),
            save_summary_steps=params["training"]['log_steps'],
            log_step_count_steps=params["training"]['log_steps'])
    else:
        raise ValueError("mode should be local or distributed")
    return tf_config

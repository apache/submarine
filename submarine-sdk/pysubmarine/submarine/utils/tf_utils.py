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
import os

import tensorflow as tf

from submarine.ml.tensorflow.optimizer import get_optimizer


def _get_session_config_from_env_var(params):
    """Returns a tf.ConfigProto instance with appropriate device_filters set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if tf_config and 'task' in tf_config and 'type' in tf_config['task'] \
            and 'index' in tf_config['task']:
        # Master should only communicate with itself and ps.
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(
                device_filters=['/job:ps', '/job:master'],
                intra_op_parallelism_threads=params["resource"]['num_thread'],
                inter_op_parallelism_threads=params["resource"]['num_thread'])
        # Worker should only communicate with itself and ps.
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(  # gpu_options=gpu_options,
                device_filters=[
                    '/job:ps',
                    '/job:worker/task:%d' % tf_config['task']['index']
                ],
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
                device_count={
                    'GPU': params["resource"]['num_gpu'],
                    'CPU': params["resource"]['num_cpu']
                },
                intra_op_parallelism_threads=params["resource"]['num_thread'],
                inter_op_parallelism_threads=params["resource"]['num_thread']),
            log_step_count_steps=params["training"]['log_steps'],
            save_summary_steps=params["training"]['log_steps'])

    elif params["training"]['mode'] == 'distributed':
        tf_config = tf.estimator.RunConfig(
            experimental_distribute=tf.contrib.distribute.DistributeConfig(
                train_distribute=tf.contrib.distribute.ParameterServerStrategy(
                ),
                eval_distribute=tf.contrib.distribute.ParameterServerStrategy(
                )),
            session_config=_get_session_config_from_env_var(params),
            save_summary_steps=params["training"]['log_steps'],
            log_step_count_steps=params["training"]['log_steps'])
    else:
        raise ValueError("mode should be local or distributed")
    return tf_config


def get_estimator_spec(logit, labels, mode, params):
    """
    Returns `EstimatorSpec` that a model_fn can return.
    :param logit: logits `Tensor` to be used.
    :param labels: Labels `Tensor`, or `dict` of same.
    :param mode: Estimator's `ModeKeys`.
    :param params: Optional dict of hyperparameters. Will receive what is passed to Estimator
     in params parameter.
    :return:
    """
    learning_rate = params["training"]["learning_rate"]
    optimizer = params["training"]["optimizer"]
    metric = params['output']['metric']

    output = tf.sigmoid(logit)
    predictions = {"probabilities": output}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(predictions)
    }
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    with tf.name_scope("Loss"):
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,
                                                    labels=labels))

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {}
    if metric == 'auc':
        eval_metric_ops['auc'] = tf.metrics.auc(labels, output)
    else:
        raise TypeError("Invalid metric :", metric)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    with tf.name_scope("Train"):
        op = get_optimizer(optimizer, learning_rate)
        train_op = op.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          train_op=train_op)

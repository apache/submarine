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
"""
Tensorflow implementation of DeepFM

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He
[2] Tensorflow implementation of DeepFM for CTR prediction
    https://github.com/ChenglongChen/tensorflow-DeepFM
[3] DeepCTR implementation of DeepFM for CTR prediction
    https://github.com/shenweichen/DeepCTR
"""

import logging

import tensorflow as tf

from submarine.ml.tensorflow.layers.core import (dnn_layer, embedding_layer,
                                                 fm_layer, linear_layer)
from submarine.ml.tensorflow.model.base_tf_model import BaseTFModel
from submarine.utils.tf_utils import get_estimator_spec

logger = logging.getLogger(__name__)


class DeepFM(BaseTFModel):

    def model_fn(self, features, labels, mode, params):
        super().model_fn(features, labels, mode, params)

        linear_logit = linear_layer(features, **params['training'])

        embedding_outputs = embedding_layer(features, **params['training'])
        fm_logit = fm_layer(embedding_outputs, **params['training'])

        field_size = params['training']['field_size']
        embedding_size = params['training']['embedding_size']
        deep_inputs = tf.reshape(embedding_outputs,
                                 shape=[-1, field_size * embedding_size])
        deep_logit = dnn_layer(deep_inputs, mode, **params['training'])

        with tf.variable_scope("DeepFM_out"):
            logit = linear_logit + fm_logit + deep_logit

        return get_estimator_spec(logit, labels, mode, params)

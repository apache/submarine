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

from submarine.ml.tensorflow.layers.core import (dnn_layer, embedding_layer, linear_layer,
                                                 KMaxPooling)
from submarine.ml.tensorflow.model.base_tf_model import BaseTFModel
from submarine.utils.tf_utils import get_estimator_spec

logger = logging.getLogger(__name__)


class CCPM(BaseTFModel):
    def model_fn(self, features, labels, mode, params):
        super().model_fn(features, labels, mode, params)

        if len(params['training']['conv_kernel_width']) != len(params['training']['conv_filters']):
            raise ValueError(
                "conv_kernel_width must have same element with conv_filters")

        linear_logit = linear_layer(features, **params['training'])
        embedding_outputs = embedding_layer(features, **params['training'])
        conv_filters = params['training']['conv_filters']
        conv_kernel_width = params['training']['conv_kernel_width']

        n = params['training']['embedding_size']
        conv_filters_len = len(conv_filters)
        conv_input = tf.concat(embedding_outputs, axis=1)

        pooling_result = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=3))(conv_input)

        for i in range(1, conv_filters_len + 1):
            filters = conv_filters[i - 1]
            width = conv_kernel_width[i - 1]
            p = pow(i / conv_filters_len, conv_filters_len - i)
            k = max(1, int((1 - p) * n)) if i < conv_filters_len else 3

            conv_result = tf.keras.layers.Conv2D(filters=filters, kernel_size=(width, 1),
                                                 strides=(1, 1), padding='same',
                                                 activation='tanh', use_bias=True, )(pooling_result)

            pooling_result = KMaxPooling(
                k=min(k, int(conv_result.shape[1])), axis=1)(conv_result)

        flatten_result = tf.keras.layers.Flatten()(pooling_result)
        deep_logit = dnn_layer(flatten_result, mode, **params['training'])

        with tf.variable_scope("CCPM_out"):
            logit = linear_logit + deep_logit

        return get_estimator_spec(logit, labels, mode, params)

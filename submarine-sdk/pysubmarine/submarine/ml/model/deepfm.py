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
import numpy as np
from submarine.ml.model.base_tf_model import BaseTFModel
from submarine.ml.layers.core import batch_norm_layer
from submarine.utils.tf_utils import get_estimator_spec

logger = logging.getLogger(__name__)


class DeepFM(BaseTFModel):
    def model_fn(self, features, labels, mode, params):
        field_size = params["training"]["field_size"]
        feature_size = params["training"]["feature_size"]
        embedding_size = params["training"]["embedding_size"]
        l2_reg = params["training"]["l2_reg"]
        batch_norm = params["training"]["batch_norm"]
        batch_norm_decay = params["training"]["batch_norm_decay"]
        seed = params["training"]["seed"]
        layers = params["training"]["deep_layers"]
        dropout = params["training"]["dropout"]

        np.random.seed(seed)
        tf.set_random_seed(seed)

        fm_bias = tf.get_variable(name='fm_bias', shape=[1],
                                  initializer=tf.constant_initializer(0.0))
        fm_weight = tf.get_variable(name='fm_weight', shape=[feature_size],
                                    initializer=tf.glorot_normal_initializer())
        fm_vector = tf.get_variable(name='fm_vector', shape=[feature_size, embedding_size],
                                    initializer=tf.glorot_normal_initializer())

        with tf.variable_scope("Feature"):
            feat_ids = features['feat_ids']
            feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
            feat_vals = features['feat_vals']
            feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

        with tf.variable_scope("First_order"):
            feat_weights = tf.nn.embedding_lookup(fm_weight, feat_ids)
            y_w = tf.reduce_sum(tf.multiply(feat_weights, feat_vals), 1)

        with tf.variable_scope("Second_order"):
            embeddings = tf.nn.embedding_lookup(fm_vector, feat_ids)
            feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
            embeddings = tf.multiply(embeddings, feat_vals)
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)
            y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)

        with tf.variable_scope("Deep-part"):
            if batch_norm:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    train_phase = True
                else:
                    train_phase = False

            deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])
            for i in range(len(layers)):
                deep_inputs = tf.contrib.layers.fully_connected(
                    inputs=deep_inputs, num_outputs=layers[i],
                    weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                    scope='mlp%d' % i)
                if batch_norm:
                    deep_inputs = batch_norm_layer(
                        deep_inputs, train_phase=train_phase,
                        scope_bn='bn_%d' % i, batch_norm_decay=batch_norm_decay)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

            y_deep = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                scope='deep_out')
            y_d = tf.reshape(y_deep, shape=[-1])

        with tf.variable_scope("DeepFM-out"):
            y_bias = fm_bias * tf.ones_like(y_d, dtype=tf.float32)
            logit = y_bias + y_w + y_v + y_d

        return get_estimator_spec(logit, labels, mode, params)
